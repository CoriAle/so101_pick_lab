# Informe: SO-101 agarra y levanta un cubo de 2.5 cm con RL (SAC)

## Tarea
El brazo SO-101 simulado (MuJoCo) debe acercarse a un cubo de 2.5 cm, agarrarlo con **ambas mandíbulas** y subir su centro **≥ 15.1 cm sobre su altura de reposo** (≈16.35 cm de Z absoluta).
El episodio termina al lograrlo, o a los 150 pasos (20 Hz).
- **Acción:** 5 incrementos de posición objetivo en [-1, 1] (pan, lift, elbow, wrist flex, gripper). `wrist_roll` está fijo.
- **Observación:** 21 valores (ver `so101_pick/observations.py`).

## Contenido de esta carpeta
| Archivo | Qué es |
|---|---|
| `best_model.zip` + `vecnormalize_best.pkl` | Política entregada y sus estadísticas de normalización (siempre juntas) |
| `rollout.mp4` | Primeros 5 episodios del comando de evaluación del ejercicio (posiciones aleatorias, 5/5 éxitos) |
| `test_*.json` | Evaluaciones de prueba por episodio |
| `selection.json` | Validación con la que se eligió el modelo |
| `config/*.json` | Configuración exacta de las 3 etapas de entrenamiento |

Evaluar: `python evaluate.py entrega/best_model.zip --episodes 100 --video rollout.mp4`.
El normalizador `vecnormalize_best.pkl` se carga solo.

## Implementación
- **`env.py`:**
  - umbral de éxito de 0.151 m;
  - currículo de reset que solo prepara la pose inicial (pinza abierta 4–10 mm sobre el cubo, que está en el suelo y sin agarrar), calculada con IK; la probabilidad baja de 1 a 0 en 50 k pasos;
  - campos extra en `info` para métricas.
- **`rewards.py`:** costos acotados con `tanh` por fase:
  - aproximación, con alineación lateral y descenso por separado;
  - apertura: penaliza cerrar antes de tiempo, no cerrar cuando ya está alineada y abrir de más;
  - agarre, con crédito parcial por contacto de una mandíbula;
  - elevación, que solo progresa mientras está agarrado;
  - penalizaciones por suelo, deslizamiento del cubo, acciones y velocidades;
  - bono de éxito `1 + pasos_restantes/150`, que premia terminar antes.
  
  Estructura, pesos y escalas tomados del trabajo de referencia de un compañero; el código es propio.
- **`callbacks.py`:**
  - currículo aplicado con `env_method`, que atraviesa los wrappers de Gym y está probado;
  - selección del mejor modelo **por tasa de éxito** (la recompensa solo desempata), siempre sobre los mismos 20 episodios.
- **`evaluate.py`:**
  - normalización emparejada y congelada, acciones deterministas;
  - fotograma terminal capturado antes del reset automático;
  - métricas por episodio en JSON.
- **Pruebas:** 38, en `tests/`. Entre otras cosas verifican:
  - agarrar sin elevar no basta;
  - elevar sin ambas mandíbulas no basta;
  - 15.0 cm no basta y 15.1 cm sí;
  - el currículo llega al entorno real y no toca el entorno de evaluación;
  - el mejor modelo se elige por éxito;
  - modelo y normalizador se guardan juntos.

## Entrenamiento (desde cero), `scripts/run_chain.sh 0 final_v1_seed0`
SAC, MLP [64, 64], lr 3e-4, batch 128, buffer 100 000, learning_starts 1 000, gamma 0.99, tau 0.005, entropía automática.
4 entornos secuenciales, 1 hilo de torch, VecNormalize solo de observaciones (recompensas sin normalizar), semilla 0.

| Etapa | `randomization` | Pasos | Inicio |
|---|---|---|---|
| 1 | 0 (cubo fijo) | 200 000 | Desde cero, currículo de reset en los primeros 50 000 |
| 2 | 0.3 | 150 000 | `best` de la etapa 1 |
| 3 | 1.0 (aleatorio completo) | 400 000 | `final` de la etapa 2 |

**Selección final** (`scripts/select_model.py`): se compararon el mejor checkpoint de la etapa 3 y el último modelo, con 100 episodios aleatorios de **validación** (semilla 500).
Esa semilla es distinta de la evaluación durante el entrenamiento (semilla 1000) y de las semillas de prueba (1 y 2).

| Candidato | Éxito en validación | Retorno medio |
|---|---|---|
| Checkpoint elegido durante el entrenamiento | 89.0 % | −38.7 |
| **Modelo final de la etapa 3 (elegido)** | **96.0 %** | −24.4 |

`best_model.zip` es por tanto una copia del modelo final de la etapa 3.

## Resultados de prueba
Política determinista, normalización congelada. Éxito = ambas mandíbulas tocan el cubo **y** está a ≥ 15.1 cm, en el mismo paso.

| Evaluación | Episodios | Éxito | Agarre (al menos una vez) | Pasos medios | Episodios con contacto brazo–suelo |
|---|---|---|---|---|---|
| Posición fija (`randomization 0`) | 50 | **100 %** | 100 % | 16.0 | 50 |
| Aleatoria, semilla 1 | 200 | 96.5 % | 100 % | 23.1 | 181 |
| Aleatoria, semilla 2 | 200 | 94.5 % | 98.5 % | 26.3 | 174 |
| **Aleatoria, total** | **400** | **95.5 %** (IC 95 % ≈ 93–97.5 %) | 99.25 % | 24.7 | 355 |
| Comando del ejercicio (`--episodes 100`, semilla 12345) | 100 | 97.0 % | 98.0 % | 22.1 | 87 |

## Historia de experimentos (resumen)
| Experimento | Qué cambió | Aleatoria (400) | Lección |
|---|---|---|---|
| E1 | Etapa 1 sola (rand 0) | 26 % (50 ep.) | Con solo el cubo fijo no generaliza |
| E1→E3 | + rand 0.3, + rand 1.0 | 96.0 % | La aleatorización creciente funciona |
| E4 | E3 + 200 k pasos | 97.25 % | Más tiempo aporta poco (+1.25 puntos) |
| E5, recompensa v2 | Agarre estable + costos por perder el agarre o empujar el cubo | 0 % (seed 0, colapso) / 78 % (seed 1) | Encarecer los fallos vuelve a la política aversa a intentar; se descartó y se eliminó del código |
| E5, v1 con semilla 1 | Solo cambia la semilla | 86.5 % | Hay mucha varianza entre semillas |
| Entrega (semilla 0) | Selección por éxito + validación final | 95.5 % | Es la que se entrega |
| Misma receta, semilla 1 | Solo cambia la semilla | 84.5 % | Confirma la varianza entre semillas |

Los modelos de E1–E5 y la semilla 1 quedan en un respaldo fuera del repositorio. E4 alcanzó un 97.25 %, pero fue una continuación de E3, mientras que la entrega es un entrenamiento desde cero con una receta reproducible.

## Limitaciones
- **No es 100 % en posiciones aleatorias.** El fallo típico es un agarre inestable: la mandíbula móvil toca primero, empuja el cubo, el agarre dura 1–2 pasos y se pierde.
- **Contacto con el suelo:** las mandíbulas tocan el suelo al cerrar en casi todos los episodios. No invalida el éxito en simulación, pero sería un problema en hardware real.
- **Posición fija:** los 50 episodios son deterministas e idénticos. Prueban que funciona en esa posición, no robustez.
- **Varianza entre semillas:** con la misma receta, la semilla 1 dio 84.5 %. El 95.5 % corresponde a la semilla 0.
- **Transparencia:** antes de escribir `select_model.py`, ya se había medido en las semillas de prueba que el modelo final superaba al checkpoint (95.5 % frente a 90.25 %). La regla de selección usa semillas de validación propias, pero esa información previa existía.
- **Solo simulación:** MuJoCo 3.10.0, Gymnasium 1.0.0, SB3 2.4.1, torch 2.2.2. Los indicadores de contacto de las mandíbulas no existen directamente en el robot real. Cambios en la física, el robot, los sensores o la observación exigen validar de nuevo.
