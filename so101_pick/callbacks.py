"""Training callbacks: reset curriculum and success-based model selection."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization


class ResetCurriculumCallback(BaseCallback):
    """Fade the probability of easy (near-cube) resets from 1 to 0 over `duration` steps.

    Only training envs are touched; the evaluation env always starts from the normal home pose.
    `env_method` is used so the call crosses Gym wrappers and reaches the real env.
    """

    def __init__(self, duration: int):
        super().__init__()
        if duration <= 0:
            raise ValueError("curriculum duration must be positive")
        self.duration = duration
        self._last = None

    def _update(self):
        probability = max(0.0, 1.0 - self.num_timesteps / self.duration)
        # push to the envs in small increments, and always when reaching 0
        if self._last is None or abs(probability - self._last) >= 0.01 or (probability == 0.0 and self._last != 0.0):
            self.training_env.env_method("set_near_cube_start_probability", probability)
            self._last = probability
        self.logger.record("curriculum/near_cube_probability", probability)

    def _on_training_start(self):
        self._update()

    def _on_step(self):
        self._update()
        return True


class SuccessSelectionEvalCallback(BaseCallback):
    """Periodic evaluation that keeps the checkpoint with the best *task success*.

    * Every evaluation replays the same cube positions (eval env reseeded with `eval_seed`),
      so checkpoints are compared on identical episodes.
    * Ranking: success rate first, mean episode reward only as tie-break.
    * The best model and its VecNormalize statistics are saved together.
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, save_dir: str,
                 eval_seed: int = 1000, stop_after_full_success: int = 0):
        super().__init__()
        self.eval_env, self.eval_freq, self.n_eval_episodes = eval_env, eval_freq, n_eval_episodes
        self.save_dir, self.eval_seed = str(save_dir), eval_seed
        self.stop_after = stop_after_full_success
        self.best_key, self.streak = None, 0
        self.history = {"timesteps": [], "success": [], "grasp": [], "reward": [], "length": []}

    def _evaluate(self):
        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_env)
        self.eval_env.seed(self.eval_seed)
        successes, grasps, rewards, lengths = [], [], [], []
        for _ in range(self.n_eval_episodes):
            obs, done, ret, n, grasped = self.eval_env.reset(), False, 0.0, 0, False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, d, infos = self.eval_env.step(action)
                done, info = bool(d[0]), infos[0]
                ret += float(r[0]); n += 1
                grasped |= bool(info.get("is_grasped", False))
            successes.append(bool(info.get("is_success", False)))
            grasps.append(grasped); rewards.append(ret); lengths.append(n)
        return float(np.mean(successes)), float(np.mean(grasps)), float(np.mean(rewards)), float(np.mean(lengths))

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        success, grasp, reward, length = self._evaluate()
        for k, v in zip(self.history, (self.num_timesteps, success, grasp, reward, length)):
            self.history[k].append(v)
        np.savez(f"{self.save_dir}/evaluations_success.npz", **{k: np.array(v) for k, v in self.history.items()})
        self.logger.record("eval/success_rate", success)
        self.logger.record("eval/grasp_rate", grasp)
        self.logger.record("eval/mean_reward", reward)
        self.logger.record("eval/mean_ep_length", length)
        key = (success, reward)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.model.save(f"{self.save_dir}/best_model")
            normalizer = self.model.get_vec_normalize_env()
            if normalizer is not None:
                normalizer.save(f"{self.save_dir}/vecnormalize_best.pkl")
            print(f"New best (success {success:.0%}, reward {reward:.1f}) at {self.num_timesteps} steps")
        self.logger.dump(self.num_timesteps)
        self.streak = self.streak + 1 if success == 1.0 else 0
        if self.stop_after and self.streak >= self.stop_after:
            print(f"Stopping: {self.streak} consecutive evaluations with 100% success.")
            return False
        return True
