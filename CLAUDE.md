# so101_pick_lab: notas del proyecto

## Propósito
Laboratorio de RL: un SO-101 simulado (MuJoCo + Gymnasium + SB3) agarra un cubo de 2.5 cm con ambas mandíbulas
y sube su centro ≥ 0.151 m sobre la altura de reposo (`LIFT_HEIGHT`, relativo; no es Z absoluta).
El episodio termina al lograrlo o a los 150 pasos (20 Hz).

## Entorno
conda `pytorch_env` (Python 3.10.18, MuJoCo 3.10.0, Gymnasium 1.0.0, SB3 2.4.1, torch 2.2.2, numpy 1.26.4).
Pruebas: `conda run -n pytorch_env python -m pytest tests`.

## Estructura
- `so101_pick/env.py`: física, contactos, éxito y currículo de reset (`set_near_cube_start_probability`; solo prepara la pose inicial).
- `so101_pick/rewards.py`: recompensa por fases con costos `tanh` (pesos tomados de la referencia de un compañero, código propio).
- `so101_pick/observations.py`: contrato de 21 valores; no cambiarlo, porque los checkpoints dependen de él.
- `so101_pick/callbacks.py`: currículo + `SuccessSelectionEvalCallback` (mejor modelo por éxito, mismos episodios de evaluación cada vez).
- `train.py`: SAC [64,64], VecNormalize solo de observaciones; guarda config y copia del código en `runs/<run>/`.
- `evaluate.py`: carga `vecnormalize_<best|final>.pkl` emparejado; JSON por episodio; fotograma terminal en el video.
- `scripts/run_chain.sh`: receta completa (rand 0 → 0.3 → 1.0, selección final y prueba). `scripts/select_model.py`: validación final (semilla 500).
- `entrega/`: modelo entregado, video, evaluaciones e informe (`REPORT.md`).

## Convenciones
- Modelo y normalizador siempre juntos.
- La posición fija y la aleatoria se reportan por separado.
- Semillas: evaluación durante el entrenamiento = 1000, validación = 500, prueba = 1 y 2. No elegir modelos usando las semillas de prueba.
- `runs/` está en `.gitignore`. Los experimentos antiguos están respaldados en `../so101_pick_lab_backup_2026-11-22/`.
