# Diagnóstico: éxito en posiciones aleatorias (E3 `final`)

## 1. Medición más precisa
Con 50 episodios E3 `final` dio 98 %. Con **400 episodios** (semillas 1 y 2) el éxito real es **96 %** (193/200 y 191/200).
El 98 % anterior era ruido de muestra.

## 2. Dónde falla
| Fase | Fallos (de 400) |
|---|---|
| No llega al cubo | 0 |
| Llega pero nunca agarra | 2 |
| **Agarra y pierde el cubo** | **14** |

- En los fallos, el agarre se pierde de media 1.75 veces por episodio; en los éxitos, 0.11.
- La posición influye poco. Con |ángulo| > 30° el éxito baja a 93 %, y los radios extremos rondan 94–96 %.
- La orientación (yaw) del cubo **no** influye, aunque `wrist_roll` está congelado.

**Mecanismo, visto en la traza de un fallo:**
1. La mandíbula móvil toca el cubo primero y lo empuja 1–2 cm.
2. Las dos mandíbulas lo tocan solo 1–2 pasos y se pierde el agarre.
3. La política reintenta, cierra mal varias veces y termina empujando el cubo unos 23 cm.
4. Con el cubo así de lejos, no vuelve a acercarse en lo que queda del episodio.

**Contacto con el suelo:** lo hacen las propias mandíbulas (`gripper` y `moving_jaw`) al cerrar muy abajo, en el 75 % de los episodios.
No es la causa principal de los fallos: algunos fallos tienen 0 pasos en el suelo.

## 3. ¿Recompensa, hiperparámetros o tiempo?

### Recompensa: causa principal
1. **Cualquier contacto momentáneo de ambas mandíbulas cuenta como agarre completo.** El costo de agarre cae a 0 aunque dure un solo paso, y **perder el agarre no tiene costo propio**. La política no aprende a asegurar la sujeción antes de subir.
2. **Empujar el cubo casi no cuesta.** El deslizamiento se penaliza con `0.1·tanh(moved/0.05)`, que satura en 0.1. Desplazar el cubo 23 cm cuesta lo mismo que 5 cm, y mucho menos que el resto de costos.
3. **Sin gradiente para recuperarse.** El costo de aproximación `0.35·tanh(lat/0.04)+0.55·tanh(|dz|/0.025)` está saturado cuando el cubo queda lejos. Lejos del cubo no hay señal que diga "vuelve".
4. **El costo de apertura castiga reabrir junto al cubo.** Con `ready≈1`, abrir la pinza aumenta `missed_close`. Esto desincentiva reabrir para reintentar tras un mal cierre.

### Hiperparámetros: causa secundaria, sobre todo en la selección del modelo
- La evaluación durante el entrenamiento usa **3 episodios**. Es demasiado ruido para elegir `best`: en E3, `best` resultó peor que `final`.
- El resto (lr, batch, red [64,64], gamma, tau) no muestra un problema claro. La política aprende la tarea en el 96 % de los casos y no hay inestabilidad.

### Tiempo de entrenamiento: aporta, pero no resuelve
- El éxito durante el entrenamiento (política estocástica) sigue subiendo lento: **0.82 → 0.95** en 200 k pasos, con oscilaciones.
- Más pasos probablemente sumen algún punto. Pero los fallos son estados raros (agarre perdido, cubo lejos) que se ven poco, y la recompensa actual no guía bien la recuperación.

## 4. Propuesta, en orden

**Paso 1: E4, solo más tiempo y mejor selección. Sin cambiar código de la tarea.**
- Continuar desde E3 `final` 200 k pasos con `randomization=1`.
- `--eval-episodes 20`, para que `best` se elija con menos ruido.
- Evaluar con 400 episodios (semillas 1 y 2).
- Esto separa el efecto del tiempo del de la recompensa.

**Paso 2: E5, recompensa con agarre estable y recuperación** (si E4 queda por debajo de ~99 %).
- *Entorno*: añadir a `state` (no a la observación, que sigue en 21) `grasp_steps`, los pasos consecutivos con agarre, y `grasp_lost`, que vale 1 en el paso en que se pierde el agarre.
- *Agarre estable*: el costo de agarre baja por grados, por ejemplo `0.3·(1 − min(grasp_steps, 3)/3)`, en vez de caer a 0 en el primer contacto.
- *Perder el agarre*: costo puntual de unos 0.3 en el paso en que se pierde.
- *Empujar el cubo*: `0.3·tanh(moved/0.03)`, más fuerte, y además un término de distancia que no sature, por ejemplo `0.1·min(dist, 0.3)/0.3`, para que haya señal de volver.
- *Reabrir*: quitar el castigo de `missed_close` cuando el cubo se ha movido más de 1 cm sin agarre, para permitir reintentar.
- *Suelo*: mantener 0.5. Si hace falta, añadir un costo pequeño cuando la pinza baja de la altura del centro del cubo.
- Requiere pruebas nuevas de recompensa:
  - perder el agarre es peor que mantenerlo;
  - empujar el cubo lejos es peor que dejarlo quieto;
  - lejos del cubo, acercarse sigue siendo mejor.
- Entrenarlo como continuación de E3 y, para validar, también desde cero (rand 0 → 0.3 → 1).
- Comprobar que la posición fija sigue en 100 %.

**Paso 3: opcional, si quedan fallos en los bordes (|ángulo| > 30°).**
Currículo que sobremuestree posiciones de borde, o más pasos con `randomization=1`.

## 4b. Diseño experimental corregido
Continuar desde E3 con una recompensa nueva contaminaría la comparación, por tres motivos:
- no se podría separar el efecto de la recompensa del de los pasos extra;
- el crítico de SAC arrastra valores Q aprendidos con la recompensa anterior;
- la política hereda los hábitos aprendidos antes.

Por eso:
- **E4 (control de tiempo):** continuación de E3 `final`, **misma recompensa v1**, 200 k pasos, `--eval-episodes 20`. Aquí continuar sí es válido, porque solo cambia el tiempo.
- **E5 (efecto de la recompensa):** **desde cero**, con el mismo calendario que E1 → E2 → E3 (`scripts/run_chain.sh`):
  200 k pasos con rand 0 y currículo, luego 150 k con rand 0.3, luego 200 k con rand 1.0. Mismos hiperparámetros y evaluación de 3 episodios.
  Lo único que cambia es la recompensa (`--reward v2`).
  - `e5_v2_seed0`, `e5_v2_seed1`: recompensa v2.
  - `e5_v1_seed1`: recompensa v1 con otra semilla, para estimar la variación entre semillas. Su par con semilla 0 es la cadena E1 → E3 original.
- **v2, tal como se implementó:**
  - agarre que crece durante 3 pasos consecutivos;
  - costo de 0.3 al perder el agarre;
  - empujar el cubo cuesta `0.3·tanh(moved/0.03)`;
  - término de distancia `0.1·min(d,0.3)/0.3` sin agarre.
  
  El cambio sobre `missed_close` se descartó. `cube_xy_moved` es acumulado desde el reset, así que habría desactivado el incentivo a cerrar también en agarres normales. La pérdida de agarre ya penaliza ese caso.

## 5. Criterio para decidir
- Evaluación aleatoria con 400 episodios, más 50 episodios en posición fija.
- Se acepta el cambio si sube el éxito aleatorio sin bajar la posición fija del 100 %, y sin aumentar el contacto con el suelo.

## 6. Resultados de E4 y E5
Modelo `final` de la última etapa. Evaluación aleatoria de 400 episodios (semillas 1 y 2) y 50 episodios en posición fija.

| Experimento | Recompensa | Semilla | Fija | Aleatoria (400) | Agarre aleatorio | Episodios aleatorios con suelo |
|---|---|---|---|---|---|---|
| E3 (cadena original) | v1 | 0 | 100 % | 96.0 % | ~99.5 % | ~75 % |
| E4 (E3 + 200 k pasos) | v1 | 0 | 100 % | **97.25 %** | 99.25 % | 80.5 % |
| E5 cadena desde cero | v1 | 1 | 100 % | 86.5 % | 95.0 % | 99.5 % |
| E5 cadena desde cero | v2 | 0 | **0 %** (colapso) | 0 % | 0 % | 2.75 % |
| E5 cadena desde cero | v2 | 1 | 100 % | 78.0 % | 92.0 % | 81.75 % |

### Conclusiones
1. **La recompensa v2 es peor y se descarta.** Con semilla 0 la política colapsó en la etapa 2 (rand 0.3): el éxito pasó de ~38 % a 0 % y la entropía cayó a 0.007.
   Con semilla 1 quedó 8.5 puntos por debajo de v1 con la misma semilla.
   Causa probable: v2 encarece los intentos fallidos (empujar el cubo 0.3, perder el agarre 0.3). Con eso, no intentar sale más barato que fallar, y la política se vuelve aversa a explorar.
2. **Error de diseño propio:** v2 cambió 4 términos a la vez, contra la regla de un cambio por ejecución del plan. No se puede saber qué término causó el daño.
3. **La varianza entre semillas es grande.** Con v1, la semilla 0 da 96 % y la semilla 1 da 86.5 %. El 96–97 % de E3/E4 depende en parte de una semilla favorable.
   Cualquier mejora futura debe medirse con al menos 2–3 semillas.
4. **Más tiempo aporta poco:** E4 suma +1.25 puntos (384 → 389 de 400), en el borde del ruido estadístico.
5. **Mejor modelo disponible:** `runs/e4_cont_time_control/final_model.zip` + `vecnormalize_final.pkl`, con 100 % en posición fija y 97.25 % en aleatoria. Es una continuación, no un entrenamiento desde cero.

### Próximos pasos propuestos (uno por vez, 2–3 semillas)
- **v2a:** solo el agarre progresivo (3 pasos consecutivos), sin los costos extra de empujar ni de perder el agarre.
- **v2b:** solo el término de distancia no saturado.
- Elegir `best` por éxito real con más episodios de evaluación. Hoy `EvalCallback` elige por recompensa media.
