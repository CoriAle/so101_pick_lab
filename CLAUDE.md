# Hallazgos de la exploración: so101_pick_lab

## Propósito
Laboratorio de RL: entrenar un brazo SO-101 simulado (MuJoCo + Gymnasium + Stable-Baselines3, sin ROS 2) para **acercarse, agarrar y levantar** un cubo de 2.5 cm. Éxito = ambas mandíbulas tocan el cubo **y** el cubo sube >= 15 cm.

## Estructura
| Ruta | Rol |
|---|---|
| `so101_pick/__init__.py` | Registra `SO101Pick-v0` (150 pasos máx.) |
| `so101_pick/env.py` | **Provisto.** Física, acciones, contactos, reset, éxito |
| `so101_pick/rewards.py` | **TODO 1 (alumno).** `compute_reward(state, action)`. Hoy es sparse: 1.0 solo si hay éxito |
| `so101_pick/observations.py` | Provisto. Observación de 21-D (no cambiarla tras entrenar: los checkpoints dependen del tamaño) |
| `train.py` | **TODO 2 (alumno).** Elegir SAC/PPO y ajustar `HPARAMS`; arnés provisto |
| `evaluate.py` | Provisto. Tasa de éxito en N episodios, video opcional (también sirve para calificar) |
| `scripts/check_env.py` | Chequeo rápido del entorno |
| `tests/test_env.py`, `tests/test_reward.py` | El primero debe pasar ya; el segundo valida el TODO 1 |
| `assets/so101/` | Modelo MuJoCo Menagerie (`so101.xml`, STL) + `pick_scene.xml` (suelo, cubo, cámaras `front` y `top`) |
| `installation.md` | Instalación local / Colab / Docker |

## Diseño del entorno (env.py)
- Control a 20 Hz (`FRAME_SKIP=10` x 5 ms). Acción 5-D en [-1,1]: pan, lift, elbow, wrist_flex, gripper. Son **deltas** sobre el objetivo de posición (`ARM_STEP=0.08` rad, `GRIPPER_STEP=0.20` rad). `wrist_roll` está congelado.
- Pose inicial `HOME_QPOS=[0,-0.5,0.4,1.5]`, gripper abierto 0.6. El cubo aparece en coordenadas polares alrededor del eje de pan: radio 18 cm ±4 cm, ángulo ±0.7 rad, escalado por `randomization` (0 = posición fija, 1 = aleatorio completo).
- `terminated` al tener éxito; `truncated` a los 150 pasos.
- `reward_fn` y `obs_fn` son inyectables; el entorno infiere el espacio de observación llamando a `obs_fn`.
- `state` expone: joint_pos/vel, ee_pos, cube_pos/quat/lin_vel, `cube_xy_moved`, `ee_to_cube`, `dist_ee_cube`, `cube_height`, `gripper_opening`, flags de contacto (`fixed_jaw_contact`, `moving_jaw_contact`, `is_grasped`, `arm_floor_contact`), `success`, `last_action`, `step`.
- `info` incluye `is_success`, que SB3 usa para la tasa de éxito.
- Las mandíbulas se detectan como geoms con `condim==6` de los cuerpos `gripper` y `moving_jaw_so101_v1`.

## Entrenamiento (train.py)
- `ALGO="SAC"` por defecto; hparams baseline SAC (lr 3e-4, batch 256, buffer 100k, [256,256]) y PPO (n_steps 2048, [64,64]).
- `USE_VECNORMALIZE=False`, `N_ENVS=4` (SubprocVecEnv). Salidas en `runs/<run-name>/` (`best_model.zip`, `final_model.zip`, logs TB, `vecnormalize.pkl` si aplica). `runs/` está en `.gitignore`.
- `--init` permite hacer fine-tuning, pero solo reaplica `learning_rate`.
- Flujo del README: A) reward inicial sparse con `--randomization 0` y ver por qué no aprende; B) reward por etapas (approach → grasp → lift → success); C) tuning de hparams con una variable por corrida y luego `--randomization 1`.

## Inconsistencias detectadas (posibles fallos de documentación)
1. **Umbral de éxito**: el código usa `LIFT_HEIGHT = 0.15` (commit "5 cm → 15 cm"), pero siguen diciendo "5 cm" `env.py` (docstring), `rewards.py` (docstring) y `evaluate.py` (mensaje impreso "lifted 5 cm").
2. **Numeración de TODOs**: `env.py` habla de TODO 1 reward, TODO 2 observaciones, TODO 3 train; el README y `observations.py` dicen que las observaciones son provistas (solo 2 TODOs).
3. **Definición de altura**: el README dice "centro del cubo >= 15 cm sobre el suelo", pero `cube_height` es relativo a la altura de reposo (centro a 1.25 cm), así que realmente se requieren ~16.25 cm de altura absoluta.
4. `train.py` enumera 3a/3b/3c (TODO 3), mientras el README lo cuenta como TODO 2.

## Puntos de atención para el reward
- El reward es "trampeable": golpear el cubo lejos, quedarse junto a él, golpear el suelo (`arm_floor_contact`), acciones temblorosas.
- Los tests exigen: float finito, acercarse mejora, agarrar > rondar, levantar agarrado mejora, éxito es lo máximo, cerrar el gripper en el aire no se recompensa, levantar sin agarre no se recompensa.
- Si la suma de recompensas por sostener el cubo supera el bono de éxito, la política puede quedarse justo bajo la altura de éxito. Se sugiere premiar rapidez (`150 - step`).
- Las banderas de contacto en la observación no son gratis en hardware real (relevante para sim-to-real).

## Estado del repo
Rama `new-attempt`, limpia. Aún no se ha ejecutado nada (ni tests ni entrenamiento); el reward sigue siendo el starter sparse.
