# Plan de acción: agarre y elevación de 15.1 cm con el SO-101

## Objetivo
- **Meta principal:** 100 % de agarre con ambas mandíbulas y 100 % de éxito en altura (centro del cubo ≥ 0.151 m sobre su altura de reposo) en la evaluación de posición fija: 50 episodios, `randomization=0`, deterministas, desde la pose normal y sin currículo.
- **Meta secundaria:** medir aparte la tasa de éxito con posiciones aleatorias (`randomization=1`). No se mezcla con la meta principal.
- Pasar las pruebas de software es requisito, pero **no demuestra aprendizaje**. Solo cuenta la evaluación con `evaluate.py`.

## Contexto
- Rama de trabajo: `new-attempt` (plantilla base). No se tocan `main`, `change-approach`, el stash ni los `runs/` existentes.
- Entorno: conda `pytorch_env`, con Python 3.10.18, MuJoCo 3.10.0, Gymnasium 1.0.0, SB3 2.4.1, torch 2.2.2, numpy 1.26.4 y 8 CPU.
  - La referencia fija MuJoCo 3.2.7. Cualquier resultado debe validarse en **este** entorno.
- Referencia: `../check/so101_pick_lab`, el entregable del compañero. Se usan su **función de recompensa** (estructura, pesos y escalas) y sus **hiperparámetros** como punto de partida. El código se escribe de forma propia, siguiendo la estructura de este repo, y el informe reconoce la referencia.
- Los assets MuJoCo y el contrato de observación (21 valores) son idénticos a los de la referencia. No se modifican.

## Fase 0: Línea base
1. Correr `scripts/check_env.py` y `tests/test_env.py` en `pytorch_env` para confirmar que el entorno funciona antes de cambiar nada.
2. Opcional (Parte A del README): un entrenamiento corto con la recompensa sparse original, para documentar por qué no aprende.

## Fase 1: Entorno (`so101_pick/env.py`)
1. `LIFT_HEIGHT = 0.151`. Corregir los docstrings que dicen "5 cm" y la numeración de TODOs.
2. Añadir a `info` los campos `arm_floor_contact` y `gripper_opening`, que necesitan las métricas.
3. Currículo de reset, que solo afecta la pose inicial y nunca las acciones:
   - Parámetro `near_cube_start_probability` (0 por defecto) y su setter público, que valida el rango [0, 1].
   - Con esa probabilidad, el reset coloca la pinza 4–10 mm sobre el cubo (±2 mm en xy), orientada hacia abajo y con apertura de 0.3 rad. El cubo sigue en el suelo, sin agarre.
   - La pose se calcula con IK por mínimos cuadrados amortiguados sobre las 4 articulaciones del brazo, respetando los límites de control.
   - Modo `interpolate` opcional, con su setter de fracción. El modo validado será `mixture`.
4. No cambiar física, acciones (0.08 / 0.20 rad), orden de articulaciones, `gripperframe` ni el horizonte de 150 pasos.

## Fase 2: Recompensa (`so101_pick/rewards.py`)
Se implementan los términos de la referencia con sus mismos pesos y escalas. Todos los costos están acotados con `tanh`.

| Término | Definición (pesos de la referencia) |
|---|---|
| Aproximación | `0.35·tanh(err_lateral/0.04) + 0.55·tanh(|dz|/0.025)`: lateral y descenso por separado |
| Disposición a cerrar | `(1−tanh(max(0,d−0.008)/0.01))·(1−tanh(max(0,err_lat−0.005)/0.01))` |
| Apertura | `0.2·(1−ready)·tanh(max(0,0.3−open)/0.2) + 0.3·ready·tanh(max(0,open−0.05)/0.6) + 0.2·tanh(max(0,open−0.6)/0.6)` |
| Agarre | Sin agarre: `0.3 + costo_apertura − 0.12·ready·(contacto de una mandíbula)`. Con agarre: 0 |
| Elevación | `0.4·(1 − tanh(clip(h/0.151,0,1))/tanh(1))`, que solo progresa mientras hay agarre |
| Penalizaciones | Constante −0.01 por paso. Deslizamiento `0.1·tanh(xy_moved/0.05)`, que se atenúa si el cubo está agarrado y en el aire. Suelo 0.5. Acción `0.01·mean(a²)`. Velocidad `0.01·mean(tanh(v/2)²)` |
| Éxito | Bono de `1 + (150−step)/150`, que reemplaza los costos de fase y premia terminar antes |

## Fase 3: Callbacks (`so101_pick/callbacks.py`, nuevo)
- `ResetCurriculumCallback`: la probabilidad baja de 1 a 0 de forma lineal en 50 000 pasos y se aplica con `env_method`, que atraviesa los wrappers de Gym. No toca el entorno de evaluación.
- `StageEvalCallback`: extiende `EvalCallback` y registra `eval/grasp_rate`, `eval/lift_rate` y `eval/max_height_mean`.
- `SaveNormalizationCallback`: guarda `vecnormalize_best.pkl` en el mismo instante que `best_model.zip`.
- `StopOnFullSuccessCallback` (opcional): detiene el entrenamiento tras N evaluaciones consecutivas con 100 % de éxito.

## Fase 4: Entrenamiento (`train.py`)
Hiperparámetros de la referencia:

| Parámetro | Valor |
|---|---|
| Algoritmo | SAC, `MlpPolicy` [64, 64] |
| learning_rate / batch / buffer | 3e-4 / 128 / 100 000 |
| learning_starts / gamma / tau | 1 000 / 0.99 / 0.005 |
| ent_coef / train_freq / gradient_steps | auto / 1 / 1 |
| Entornos | 4 con `DummyVecEnv` (secuencial), 1 hilo de torch |
| Normalización | VecNormalize solo de observaciones; recompensas sin normalizar |
| Semilla / randomization | 0 / 0 |
| Evaluación | Cada 5 000 pasos, 3 episodios deterministas desde la pose normal |
| Currículo | `mixture`, 50 000 pasos |
| Pasos totales | 200 000 |

- Opciones de línea de comandos para cada valor anterior, más `--init`, `--vecnormalize` y `--replay-buffer` para continuar un entrenamiento.
- Cada ejecución guarda en `runs/<nombre>/`:
  - `config.json`.
  - Copias de `env.py`, `rewards.py`, `observations.py`, `callbacks.py` y `train.py`.
  - `best_model.zip` + `vecnormalize_best.pkl` y `final_model.zip` + `vecnormalize_final.pkl`, siempre en pares.
- Al interrumpir con Ctrl+C se guarda el estado actual. Nunca se sobrescriben ejecuciones anteriores.

## Fase 5: Evaluación (`evaluate.py`)
- Carga automáticamente el `vecnormalize_<best|final>.pkl` que corresponde al modelo. La normalización se congela (`training=False`, `norm_reward=False`) y las acciones son deterministas.
- Métricas por episodio, guardadas en JSON:
  - éxito, agarre y elevación máxima mientras está agarrado;
  - distancia mínima, duración, pasos con el brazo en el suelo y retorno.
- Un wrapper captura el fotograma terminal antes del reset automático, para que el video muestre el final real.
- Los resultados de posición fija y aleatoria se guardan en archivos separados.

## Fase 6: Pruebas (`tests/`)
- **Entorno:**
  - el éxito exige ambas mandíbulas y h ≥ 0.151;
  - 0.150 m no basta;
  - agarrar sin elevar no basta;
  - elevar con una sola mandíbula no basta;
  - el éxito en el paso 150 cuenta como `terminated`;
  - el reset fácil deja la pinza cerca, abierta, con el cubo en el suelo y sin agarre;
  - con probabilidad 0 se recupera el reset estándar;
  - se conservan las pruebas originales.
- **Recompensa:** las pruebas originales, más las de la referencia:
  - apertura durante la aproximación y cierre al estar alineado;
  - bajar es mejor que flotar;
  - soltar el cubo empeora la recompensa;
  - la altura sin agarre no cuenta;
  - esperar bajo la meta cuesta;
  - el éxito temprano es mejor;
  - el suelo, el deslizamiento y el movimiento excesivo se penalizan.
- **Callbacks:**
  - el currículo llega al entorno real a través de los wrappers y se desvanece;
  - el entorno de evaluación no cambia;
  - el mejor checkpoint guarda sus propias estadísticas;
  - la parada por éxito funciona.
- **Evaluación:** el fotograma terminal se conserva tras el reset automático.

## Fase 7: Experimentos hacia el 100 %
1. **E1, réplica de la configuración de referencia:** 200 k pasos, semilla 0, `randomization=0`.
   - Evaluar `best` y `final` con 50 episodios en posición fija.
2. **Criterio de aceptación:** agarre 50/50, éxito 50/50 y 0 episodios con contacto del brazo con el suelo.
3. **Si E1 no llega al 100 %:**
   - Diagnosticar la fase que falla con las métricas por fase y el video: aproximación, cierre, retención o elevación.
   - Probar estas variantes, cambiando **una sola cosa** por ejecución y con un `--run-name` nuevo cada vez:
     - **E2:** más pasos (300–400 k), o continuar desde el mejor modelo con su normalizador y su replay buffer, etiquetado como continuación.
     - **E3:** semillas 1 y 2, para descartar que el resultado dependa de la semilla.
     - **E4:** más episodios por evaluación (10–20) para que la selección del mejor modelo sea más estable, o parada tras 3 evaluaciones consecutivas al 100 %.
     - **E5:** ajustes de recompensa solo si el diagnóstico apunta a un término concreto. Por ejemplo: subir el peso de elevación si se queda bajo los 15.1 cm, o el del costo de cierre si no cierra.
4. Elegir el modelo final por éxito real en la evaluación de 50 episodios, no por la recompensa. Se conservan todos los modelos.
5. **Generalización (secundaria):** evaluar el modelo elegido con `randomization=1` (50 episodios, semilla distinta). Opcionalmente, continuar el entrenamiento con `randomization` 0.3, luego 1.0, y reportar cada etapa por separado.

## Fase 8: Entregables
- `runs/<ejecución_final>/REPORT.md` y `origin.json`, con:
  - hashes del checkpoint y del normalizador;
  - origen: "desde cero" o "continuación";
  - referencia usada.
- Evaluaciones fija y aleatoria en JSON y video `rollout.mp4` con el fotograma terminal.
- README y `CLAUDE.md` actualizados con el flujo, el umbral de 15.1 cm y las versiones del entorno.
- **Limitaciones a declarar:**
  - validado solo en simulación, con MuJoCo 3.10.0;
  - los 50 episodios fijos son deterministas e idénticos, así que no prueban generalización;
  - los indicadores de contacto no existen gratis en hardware real;
  - cualquier cambio de física, robot, sensores u observación exige una validación nueva.

## Orden de ejecución
Fase 0, luego 1, 2 y 3, con las pruebas de la Fase 6 escritas junto a cada una. Después 4 y 5, se corre toda la suite de pruebas, y siguen E1, el diagnóstico, iteraciones si hacen falta, y por último el informe.
