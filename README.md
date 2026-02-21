# DQN Robot Navigation — TurtleBot3

Sistema de navegación autónoma para TurtleBot3 Burger usando Deep Q-Network (DQN). El agente aprende a alcanzar objetivos generados aleatoriamente en un mundo de simulación Gazebo (gz), evitando obstáculos usando únicamente datos de LiDAR y odometría — sin mapa, sin planificador global.

---

## Tabla de Contenidos

- [Arquitectura general](#arquitectura-general)
- [Diseño del estado](#diseño-del-estado)
- [Espacio de acciones](#espacio-de-acciones)
- [Función de recompensa](#función-de-recompensa)
- [Hiperparámetros](#hiperparámetros)
- [Curvas de entrenamiento y resultados](#curvas-de-entrenamiento-y-resultados)
- [Análisis de resultados](#análisis-de-resultados)
- [Instalación y ejecución](#instalación-y-ejecución)
- [VIDEOS DE DEMOSTRACION](#Videos-de-Demostracion)


---

## Arquitectura general

```
Simulación Gazebo (TurtleBot3 Burger)
        │  /scan  (LaserScan)
        │  /odom  (Odometry)
        ▼
  TurtleBot3Env          ← Nodo ROS 2 — encapsula I/O del simulador, lógica de step y resets
        │
  StateProcessor         ← Agrupa el LiDAR en 10 sectores + features de navegación
        │
  DQNAgent               ← Double DQN implementado con sklearn MLPRegressor
  (256 → 256 → 128)         Experience replay + target network periódico
        │
  train_node / test_node ← Bucle de episodios, reset por outcome, logging, gráficas
```

La Q-network usa `sklearn.neural_network.MLPRegressor` con `warm_start=True` y `partial_fit` para aprendizaje online incremental. Una **target network** se mantiene como copia profunda y se sincroniza cada `target_update_freq` pasos de replay para estabilizar el entrenamiento (Double DQN).

---

## Diseño del estado

**Vector de estado — 12 dimensiones**

| Índice | Feature | Fuente | Descripción |
|--------|---------|--------|-------------|
| 0–9 | `lidar_bins[0..9]` | `/scan` | Escaneo LiDAR 360° comprimido en 10 bins angulares iguales con **min-pooling** (36 rayos/bin). Rango [0, 3.5] m. `Inf` → 3.5 m · `NaN` → 0.0 m |
| 10 | `goal_distance` | `/odom` | Distancia euclidiana al objetivo (m), calculada en `odom_callback` |
| 11 | `goal_angle` | `/odom` | Ángulo relativo al objetivo respecto al heading del robot (rad), normalizado a [−π, π] |

---

## Espacio de acciones

**7 acciones discretas**

| ID | Vel. lineal (m/s) | Vel. angular (rad/s) | Descripción |
|----|------------------|----------------------|-------------|
| 0 | 0.26 | 0.00 | Avanzar |
| 1 | 0.00 | +0.75 | Girar izquierda |
| 2 | 0.00 | −0.75 | Girar derecha |
| 3 | 0.18 | +0.50 | Avanzar + izquierda |
| 4 | 0.18 | −0.50 | Avanzar + derecha |
| 5 | 0.00 | +1.50 | Girar izquierda rápido |
| 6 | 0.00 | −1.50 | Girar derecha rápido |

Las acciones 1, 2, 5 y 6 se clasifican como **pure rotations** y son monitoreadas por las penalizaciones de oscilación y rotación en la función de recompensa.

---

## Función de recompensa

Diseñada para priorizar el movimiento exploratorio hacia adelante manteniendo un comportamiento seguro. **No existe penalización por paso de tiempo** — un time penalty enseña al agente a girar en su propio eje (la forma más rápida de terminar un episodio) en lugar de navegar.

### Componentes por paso (no terminales)

| Componente | Fórmula | Rango efectivo | Justificación |
|------------|---------|----------------|---------------|
| Progreso de distancia | `(d_prev − d_curr) × 15` | ilimitado | Señal densa principal — cada centímetro ganado hacia el goal se recompensa |
| Alineación de yaw | `(1 − 2·\|θ\|/π) × 0.5` | [−0.5, +0.5] | Guía de orientación suave; peso reducido para que no domine |
| Proximidad a obstáculos | decay direccional si algún rayo frontal < 0.30 m | [−0.9, 0] | Solo activo en rango de peligro real; ponderación direccional coseno⁶ |
| Forward bonus | +0.2 (lineal > 0) · −0.1 (rotación pura) | {−0.1, +0.2} | Incentivo pequeño a usar acciones que trasladan |
| Penalización de oscilación | −1.0 si los últimos 3 pasos forman A→B→A con (A,B) par de inversión | {−1.0, 0} | Rompe bucles de oscilación izquierda-derecha |
| Penalización de rotación | −0.5 si los últimos 5 pasos son todos rotaciones puras | {−0.5, 0} | Evita girar indefinidamente en el mismo lugar |

### Recompensas terminales

| Evento | Condición de disparo | Recompensa |
|--------|---------------------|------------|
| **Goal alcanzado** | `goal_distance < 0.20 m` | +200.0 |
| **Colisión** | > 3 rayos LiDAR < 0.25 m | −50.0 |

### Terminación del episodio y lógica de reset

| Condición | Reset |
|-----------|-------|
| Colisión | `reset_full()` — robot teletransportado al origen (0, 0) + nuevo goal aleatorio |
| Goal alcanzado | `reset_goal_only()` — robot mantiene posición + nuevo goal |
| Timeout (700 pasos) | `reset_goal_only()` — robot mantiene posición + nuevo goal |

Solo una **colisión** activa el respawn completo del robot. Los resets por timeout o goal únicamente regeneran el marker del objetivo, manteniendo al robot en su posición para preservar la continuidad del episodio.

---

## Hiperparámetros

### DQN Agent

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `state_size` | 12 | Dimensión del input |
| `action_size` | 7 | Dimensión del output |
| `learning_rate` | 0.0003 | Optimizador Adam |
| `gamma` | 0.99 | Factor de descuento |
| `epsilon_start` | 1.0 | Exploración total al inicio |
| `epsilon_min` | 0.05 | Exploración residual mínima |
| `epsilon_decay_steps` | 6 000 | Pasos de decay exponencial |
| `memory_size` | 20 000 | Buffer de experience replay (FIFO) |
| `batch_size` | 64 | Muestras por actualización |
| `target_update_freq` | 200 | Pasos de replay entre sincronizaciones de la target network |

**Esquema de decay de epsilon**
```
ε(t) = ε_min + (1 − ε_min) · exp(−t / 6000)
```

| Pasos de replay (t) | ε |
|--------------------|---|
| 0 | 1.000 |
| 2 000 | 0.385 |
| 4 000 | 0.101 |
| 6 000 | 0.050 (mínimo) |

### Arquitectura de la red

```
Input (12) → FC 256 ReLU → FC 256 ReLU → FC 128 ReLU → Output (7 Q-values)
```

Regularización L2 `α = 0.0001`. Optimizador: Adam. Entrenamiento incremental via `MLPRegressor.partial_fit`.

### Bucle de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Total de episodios | 220 |
| Pasos máximos por episodio | 700 |
| Inicio del replay | tras 64 transiciones en el buffer |
| Frecuencia de replay | en cada paso (una vez buffer ≥ batch_size) |
| Checkpoints guardados | ep 25, 50, 75, 100, 150, 200, final |

---

## Curvas de entrenamiento y resultados

<img src="https://github.com/VerabelGonzales/dqn_navigation_ws/blob/main/models/results_20260219_180800/training_results.png" alt="Acoples">
<p style="margin-top:10px; font-size: 16px;"><strong>Figura 1.</strong> Acoples</p>
<br>

### Descripción panel por panel

**Arriba izquierda — Episode Rewards:** Recompensa bruta por episodio. La alta varianza es esperada con exploración ε-greedy. Las recompensas tienden a ser positivas desde ~ep 50, lo que indica que el agente empieza a alcanzar goals más que a acumular penalizaciones por colisión.

**Arriba derecha — Moving Average (ventana = 20):** Confirma una tendencia de aprendizaje clara: cerca de cero en ep 0, subiendo a ~150–200 hacia ep 75, con una caída a mitad del entrenamiento (ep 125–150) cuando epsilon baja lo suficiente para que el agente empiece a explotar una política parcialmente convergida — una ventana de inestabilidad conocida en DQN.

**Centro izquierda — Steps per Episode:** Los episodios tempranos frecuentemente alcanzan el límite de pasos (línea punteada). A partir de ep 100, los episodios terminan antes, lo que indica que el agente toma decisiones concretas en lugar de deambular.

**Centro derecha — Episode Outcomes (% acumulado):**

| Fase | Outcome dominante | Interpretación |
|------|------------------|----------------|
| ep 0–30 | Timeout (~80%) | Política aleatoria raramente encuentra el goal |
| ep 30–80 | Colisiones suben a ~60% | El agente actúa con intención pero sin precisión |
| ep 80–220 | Éxitos crecen a ~45% | La política madura; timeouts casi eliminados |

**Abajo izquierda — Final Distance to Goal:** A partir de ep 100 la distribución se concentra bajo 0.5 m. Los picos esporádicos corresponden a goals generados lejos de la posición final del robot.

**Abajo derecha — Min Distance Achieved:** El agente llega de manera consistente a menos de 0.5 m del objetivo desde ep 80+, y frecuentemente cruza el umbral de 0.20 m (línea punteada verde), demostrando comportamiento de aproximación estable.

### Métricas resumen — fin del entrenamiento (ep 220)

| Métrica | Valor |
|---------|-------|
| Tasa de éxito | ~45% |
| Tasa de colisiones | ~45% |
| Tasa de timeouts | ~10% |
| Recompensa media móvil (últimos 20 ep) | ~200 |

---

## Análisis de resultados

### Qué logró el entrenamiento

Tras 220 episodios el agente alcanza una **tasa de éxito de ~45%** — un resultado significativo considerando que el entorno usa placement de goals completamente aleatorio en un mundo de 4×4 m con obstáculos estáticos, y que la Q-network está construida sobre `sklearn MLPRegressor` en lugar de un framework de deep learning con aceleración GPU.

### Tres fases de aprendizaje

Las curvas revelan tres fases consistentes con la teoría de DQN:

**Fase 1 — Exploración pura (ep 0–30).** ε > 0.8. El agente se mueve mayoritariamente de forma aleatoria. Timeout domina porque las acciones aleatorias rara vez construyen un camino coherente hacia el goal.

**Fase 2 — Transición (ep 30–100).** ε cae de ~0.8 a ~0.1. La política empieza a formarse pero es poco confiable. La tasa de colisiones alcanza su pico aquí: el agente ahora se mueve con intención pero todavía no tiene la precisión para esquivar obstáculos mientras persigue el objetivo. Esta es la ventana de aprendizaje crítica — la señal de distance progress acumula suficiente experiencia para moldear los Q-values hacia comportamiento dirigido al goal.

**Fase 3 — Explotación (ep 100–220).** ε ≈ 0.05. El agente actúa mayoritariamente de forma greedy. La tasa de éxito crece de manera sostenida. Los timeouts casi desaparecen (< 10%), lo que significa que el agente siempre llega a un resultado definitivo — goal u obstáculo — en lugar de deambular. La tasa de colisiones estabilizándose en ~45% refleja toma de riesgo intencional, que es el comportamiento deseado dada la penalización de colisión reducida.

### Decisiones de diseño y sus efectos observados

| Decisión | Justificación | Efecto observado |
|----------|---------------|-----------------|
| Distance progress × 15 como señal dominante | Recompensa densa en cada paso | El agente se acerca activamente al goal desde ep 30+ |
| Sin penalización de tiempo | Evita el óptimo local de "quedarse quieto" | Timeouts eliminados hacia ep 100 |
| Penalización por colisión −50 (no −100) | Evita parálisis en el entrenamiento temprano | El agente explora de forma agresiva; colisiones normales al inicio |
| Penalización de obstáculos solo < 0.30 m | Evita penalizar pasos cercanos legítimos | El robot navega espacios estrechos sin comportamiento excesivamente conservador |
| `reset_full()` solo en colisión | Preserva diversidad de posiciones | El agente aprende desde muchas posiciones, no solo desde el origen |

### Limitaciones y mejoras potenciales

| Limitación | Mejora sugerida |
|-----------|----------------|
| ~45% de colisiones persiste | Entrenar por 500+ episodios; la curva sigue subiendo en ep 220 |
| Backend `sklearn` es solo CPU | Reemplazar con PyTorch para redes más grandes y aceleración GPU |
| Buffer de 20 000 muestras puede ser pequeño | Aumentar a 100 000; las experiencias tempranas se desplazan demasiado rápido |
| Mundo de obstáculos fijo | Agregar curriculum: empezar con espacio abierto, añadir obstáculos progresivamente |

---

## Instalación y ejecución

### Requisitos del sistema

| Componente | Versión |
|-----------|---------|
| OS | Ubuntu 24.04 |
| ROS 2 | **Jazzy** |
| Simulador | **Gazebo (gz-sim)** |
| Python | 3.10+ |
| Modelo TurtleBot3 | Burger |

### Dependencias Python

```bash
pip install numpy scikit-learn matplotlib
```

### Paquetes ROS 2 y TurtleBot3

```bash
sudo apt install \
  ros-jazzy-turtlebot3 \
  ros-jazzy-turtlebot3-msgs \
  ros-jazzy-turtlebot3-simulations
```
---

### Ejecución paso a paso

#### 1. Compilar el paquete

```bash
cd ~/dqn_navigation_ws
colcon build --packages-select dqn_robot_nav
source install/setup.bash
```

#### 2. Lanzar el mundo en Gazebo

```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage2.launch.py
```

#### 3. Ejecutar el entrenamiento

```bash
# Terminal nueva
source ~/dqn_navigation_ws/install/setup.bash
ros2 run dqn_robot_nav train_node
```

Los resultados se guardan automáticamente en:
```
results_YYYYMMDD_HHMMSS/
├── training_log.txt        ← CSV por episodio: Episode, Steps, Reward, Outcome, Distance, Epsilon
├── final_statistics.txt    ← métricas resumen al final del entrenamiento
├── training_results.png    ← gráficas de entrenamiento (6 paneles)
├── model_ep25.pkl          ┐
├── model_ep50.pkl          │  checkpoints periódicos
├── model_ep75.pkl          │
├── model_ep100.pkl         │
├── model_ep150.pkl         │
├── model_ep200.pkl         ┘
└── model_final.pkl         ← usar este para evaluación
```

#### 4. Ejecutar la evaluación

```bash
source ~/dqn_navigation_ws/install/setup.bash
ros2 run dqn_robot_nav test_node --ros-args \
  -p model_path:=/ruta/absoluta/a/results_YYYYMMDD_HHMMSS/model_final.pkl
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `model_path` | *(obligatorio)* | Ruta absoluta al archivo `.pkl` del modelo |
| `n_episodes` | 10 | Número de episodios de evaluación |
| `max_steps` | 1500 | Límite de pasos por episodio |
| `stage` | 2 | Stage del mundo Gazebo |

Salida (Datos optenidos del test realizado):
```
[INFO] ====================================================
[INFO] === RESULTADOS DE EVALUACIÓN ========================
[INFO]   Episodios    : 10
[INFO]   ✓ Éxitos     : 9  (90.0%)
[INFO]   ✗ Colisiones : 0  (0.0%)
[INFO]   ⏱ Timeouts   : 1  (10.0%)
[INFO]   Reward medio : 539.17
[INFO]   Reward std   : 193.27
```

#### 5. Monitorear tópicos (opcional)

```bash
ros2 topic echo /cmd_vel        # comandos de velocidad enviados al robot
ros2 topic echo /goal_pose      # posición actual del objetivo
ros2 topic echo /scan           # datos brutos del LiDAR
ros2 topic echo /odom           # odometría del robot
```

---

## Videos de Demostracion

### Train, velocidad x16

[![Ver Video](https://github.com/VerabelGonzales/dqn_navigation_ws/blob/main/models/results_20260219_180800/miniatura.jpg)](https://www.youtube.com/watch?v=WpRYxxFKkw0)


### Test, velocidad x1 (NO X5)

[![Ver Video](https://github.com/VerabelGonzales/dqn_navigation_ws/blob/main/models/results_20260219_180800/miniatura.jpg)](https://www.youtube.com/watch?v=cgK1zQAt3Ns)