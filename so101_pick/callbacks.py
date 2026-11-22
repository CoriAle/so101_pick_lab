"""Training callbacks: reset curriculum, per-stage evaluation metrics, paired normalisation stats."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


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


class SaveNormalizationCallback(BaseCallback):
    """Used as `callback_on_new_best`: stores the VecNormalize stats next to best_model.zip."""

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def _on_step(self):
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is not None:
            normalizer.save(self.path)
        return True


class StageEvalCallback(EvalCallback):
    """EvalCallback that also logs grasp rate, lift rate and mean peak height per evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._open, self._done = {}, []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._open, self._done = {}, []
        return super()._on_step()

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        info, idx = locals_["info"], locals_.get("i", 0)
        ep = self._open.setdefault(idx, [False, False, 0.0])   # grasped, success, peak height
        ep[0] |= bool(info.get("is_grasped", False))
        ep[1] |= bool(info.get("is_success", False))
        ep[2] = max(ep[2], float(info.get("cube_height", 0.0)))
        if locals_["done"]:
            self._done.append(self._open.pop(idx))
            grasp, lift, height = np.mean(self._done, axis=0)
            self.logger.record("eval/grasp_rate", float(grasp))
            self.logger.record("eval/lift_rate", float(lift))
            self.logger.record("eval/max_height_mean", float(height))


class StopOnFullSuccessCallback(BaseCallback):
    """Used as `callback_after_eval`: stop after N consecutive evaluations with 100% success."""

    def __init__(self, evaluations: int):
        super().__init__()
        self.evaluations, self.streak = evaluations, 0

    def _on_step(self):
        buf = self.parent._is_success_buffer
        self.streak = self.streak + 1 if buf and all(buf) else 0
        if self.streak >= self.evaluations:
            print(f"Stopping: {self.streak} consecutive evaluations with 100% success.")
            return False
        return True
