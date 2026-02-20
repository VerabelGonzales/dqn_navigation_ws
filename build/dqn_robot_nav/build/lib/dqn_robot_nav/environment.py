import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from typing import Tuple
import math
import subprocess
import time


class TurtleBot3Env(Node):
    """ROS2 Environment wrapper for TurtleBot3 navigation"""

    # Distancia minima valida entre robot y objetivo al hacer spawn
    MIN_GOAL_DISTANCE = 1.0   # metros
    MAX_GOAL_RETRIES  = 50    # intentos maximos para encontrar posicion valida
    # Pasos minimos tras reset antes de poder declarar goal
    # Evita falso positivo cuando el robot spawna cerca del nuevo objetivo
    MIN_STEPS_BEFORE_GOAL = 5

    def __init__(self, stage_num: int = 1):
        super().__init__('turtlebot3_env')
        self.stage = int(stage_num)

        # Publishers and Subscribers
        self.cmd_vel_pub  = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        # Publisher de goal_pose: fuente única de verdad para visualización y logging.
        # Cualquier herramienta (RViz, scripts externos) puede suscribirse aquí
        # y ver exactamente el mismo goal que usa el código de detección.
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan',
                                                  self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                  self.odom_callback, 10)

        # State variables
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0

        # Calculados en odom_callback (igual que dqn_environment.py)
        self.goal_angle    = 0.0
        self.goal_distance = 1.0

        # Front ranges para obstacle reward direccional
        self.front_ranges = []
        self.front_angles = []
        self.min_obstacle_distance = 10.0

        # Goal position
        self.goal_position = (1.5, 0.0)

        # Goal marker entity
        self.entity_name        = 'goal_box'
        self.entity_model_path  = None
        self.entity_spawned     = False
        self._load_entity_sdf()

        # Nombre del modelo en Gazebo (confirmado por dqn_gazebo.py de referencia).
        # El SDF está en models/turtlebot3_burger/ pero el nombre de instancia es 'burger'.
        self.robot_name = 'burger'

        # -------------------------------------------------------
        # Action space: 7 acciones (SIN movimiento hacia atras)
        # Velocidades lineales aumentadas para que el robot avance
        # con intención en lugar de moverse tímidamente.
        # -------------------------------------------------------
        self.actions = {
            0: (0.26,  0.0),   # Forward rápido
            1: (0.0,   0.75),  # Rotate left
            2: (0.0,  -0.75),  # Rotate right
            3: (0.18,  0.50),  # Forward + left
            4: (0.18, -0.50),  # Forward + right
            5: (0.0,   1.50),  # Rotate left fast
            6: (0.0,  -1.50),  # Rotate right fast
        }

        self.PURE_ROTATIONS    = {1, 2, 5, 6}
        self.BACKWARD_ACTIONS  = set()
        self.OSCILLATION_PAIRS = [
            (1, 2), (2, 1),
            (3, 4), (4, 3),
            (5, 6), (6, 5),
        ]

        # Umbrales
        self.collision_threshold = 0.25
        self.goal_threshold      = 0.30

        self.previous_action = None
        self.action_history  = []
        self.max_action_history = 6
        self.last_distance = 1.0
        self.episode_step  = 0   # contador de pasos dentro del episodio actual

        # Flag para evitar doble reset: el step solo reporta done, el reset
        # lo maneja EXCLUSIVAMENTE el train_node segun el tipo de terminacion.
        self._last_done_reason = None   # 'collision' | 'goal' | 'timeout'

    # -------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        num_rays  = len(msg.ranges)
        self.scan_data    = []
        self.front_ranges = []
        self.front_angles = []

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        for i in range(num_rays):
            angle    = angle_min + i * angle_inc
            distance = msg.ranges[i]
            if distance == float('Inf') or distance > 3.5:
                distance = 3.5
            elif math.isnan(distance):
                distance = 0.0
            self.scan_data.append(distance)

            # Sector frontal: +-90 grados
            if (0 <= angle <= math.pi / 2) or (3 * math.pi / 2 <= angle <= 2 * math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        if self.scan_data:
            self.min_obstacle_distance = min(self.scan_data)

    def odom_callback(self, msg: Odometry):
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z +
                         orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y +
                              orientation_q.z * orientation_q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        self.goal_distance = math.sqrt(dx ** 2 + dy ** 2)

        path_theta  = math.atan2(dy, dx)
        goal_angle  = path_theta - self.yaw
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi
        self.goal_angle = goal_angle

    # -------------------------------------------------------
    # Step — NO hace ningun reset internamente.
    # Solo informa done=True y registra la razon en _last_done_reason.
    # El reset lo delega completamente al train_node.
    # -------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Ejecuta una accion. Solo detecta colisiones internamente.
        La deteccion de goal se delega completamente al train_node,
        que usa goal_position_frozen (variable local congelada al inicio
        del episodio) para evitar cualquier race condition con callbacks.
        """
        self.episode_step += 1
        linear_vel, angular_vel = self.actions[action]
        self.send_velocity(linear_vel, angular_vel)
        rclpy.spin_once(self, timeout_sec=0.05)

        if self.is_collision():
            reward = -50.0   # reducido de -100 para no paralizar el aprendizaje temprano
            self._last_done_reason = 'collision'
            self.get_logger().info("Collision detected!")
            self.action_history.append(action)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)
            self.previous_action = action
            return self.get_state(), reward, True   # done=True solo por colision

        # Sin colision: calcular reward y devolver done=False.
        # El train_node decidira si es goal o continua.
        reward = self.compute_reward(action)
        self._last_done_reason = None

        self.action_history.append(action)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)
        self.previous_action = action

        return self.get_state(), reward, False

    # -------------------------------------------------------
    # Reward
    # -------------------------------------------------------
    def compute_reward(self, action: int) -> float:
        """
        Función de recompensa rebalanceada para fomentar exploración:

          1. distance_progress : señal densa principal (+/- según avance).
                                 Escalado x15 — domina sobre el resto.
          2. yaw_reward        : orientación al objetivo, peso x0.5.
                                 Guía suave sin paralizar al robot.
          3. obstacle_reward   : penalización SOLO si <0.30m, peso x0.3.
                                 Rango real [-0.9, 0]. No domina al progress.
          4. forward_bonus     : incentivo leve a avanzar vs girar puro.
          5. oscillation_penalty: penaliza izq-der-izq en 3 pasos.
          6. rotation_penalty  : penaliza >4 giros puros seguidos.

        SIN time_penalty: no castigar al robot por "tardar". Con epsilon
        alto y pocas colisiones vistas, el time_penalty solo enseña a
        quedarse quieto o girar en círculos lo más rápido posible.
        """

        # 1. DISTANCE PROGRESS — señal densa y dominante
        current_dist      = self.goal_distance
        prev_dist         = self.last_distance
        distance_progress = (prev_dist - current_dist) * 15.0
        self.last_distance = current_dist

        # 2. YAW REWARD — guía suave de orientación, peso reducido
        yaw_reward = (1.0 - (2.0 * abs(self.goal_angle) / math.pi)) * 0.5

        # 3. OBSTACLE REWARD — solo <0.30m, muy suave
        obstacle_reward = self._compute_weighted_obstacle_reward() * 0.3

        # 4. FORWARD BONUS — pequeño incentivo a usar acciones con avance
        linear_vel    = self.actions[action][0]
        forward_bonus = 0.2 if linear_vel > 0.0 else -0.1

        # 5. OSCILLATION PENALTY — izq-der-izq en 3 pasos
        oscillation_penalty = 0.0
        if len(self.action_history) >= 3:
            a, b, c = self.action_history[-3], self.action_history[-2], self.action_history[-1]
            if (a, b) in self.OSCILLATION_PAIRS and c == a:
                oscillation_penalty = -1.0

        # 6. ROTATION PENALTY — >4 rotaciones puras seguidas
        rotation_penalty = 0.0
        if len(self.action_history) >= 5:
            last5 = self.action_history[-5:]
            if all(a in self.PURE_ROTATIONS for a in last5):
                rotation_penalty = -0.5

        total_reward = (distance_progress
                        + yaw_reward
                        + obstacle_reward
                        + forward_bonus
                        + oscillation_penalty
                        + rotation_penalty)
        return total_reward

    def _compute_weighted_obstacle_reward(self) -> float:
        """
        Penalización direccional de obstáculos.
        Umbral: 0.30m — solo penaliza obstáculos realmente peligrosos.
        Peso externo en compute_reward: x0.3, rango efectivo [-0.9, 0].
        No domina sobre la señal de progreso de distancia.
        """
        if not self.front_ranges or not self.front_angles:
            return 0.0

        front_ranges = np.array(self.front_ranges)
        front_angles = np.array(self.front_angles)

        # Solo obstáculos muy cercanos (<0.30m)
        valid_mask = front_ranges <= 0.30
        if not np.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = np.unwrap(front_angles)
        relative_angles[relative_angles > np.pi] -= 2 * np.pi

        weights    = self._compute_directional_weights(relative_angles)
        safe_dists = np.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay      = np.exp(-2.0 * safe_dists)

        return -(1.0 + 2.0 * np.dot(weights, decay))  # rango base [-3, 0]

    def _compute_directional_weights(self, relative_angles: np.ndarray,
                                      max_weight: float = 10.0) -> np.ndarray:
        raw_weights = np.cos(relative_angles) ** 6 + 0.1
        scaled      = raw_weights * (max_weight / np.max(raw_weights))
        return scaled / np.sum(scaled)

    # -------------------------------------------------------
    # Termination checks
    # -------------------------------------------------------
    def is_collision(self) -> bool:
        if self.scan_data is None:
            return False
        scan_array    = np.array(self.scan_data)
        scan_array    = scan_array[~np.isinf(scan_array)]
        if len(scan_array) == 0:
            return False
        return int(np.sum(scan_array < self.collision_threshold)) > 3

    def is_goal_reached(self) -> bool:
        """Wrapper para compatibilidad; usa snapshot interno."""
        return self._is_goal_reached_safe(self.position, self.goal_position)

    def _is_goal_reached_safe(self,
                               pos_snap: Tuple[float, float],
                               goal_snap: Tuple[float, float]) -> bool:
        """
        Verificación robusta de llegada al objetivo.
        Requiere TRES condiciones simultáneas para evitar falsos positivos:

        1. Distancia en tiempo real (pos_snap vs goal_snap) < threshold.
           Usa snapshots atómicos tomados justo después del spin_once,
           no self.position/self.goal_position que pueden cambiar por callbacks.

        2. self.goal_distance (calculado en odom_callback con la misma
           goal_position) también < threshold.
           Actúa como segunda fuente independiente: si el odom_callback
           vio una distancia pequeña con el goal ANTERIOR (antes de un reset),
           este check fallará porque goal_distance ya se recalculó con el nuevo goal.

        3. Mínimo de pasos transcurridos desde el último reset (self.episode_step).
           Evita el falso positivo inmediato cuando el robot spawna cerca del
           nuevo objetivo generado justo después de una colisión.
        """
        # Condición 1: distancia real con snapshots atómicos
        real_dist = self._dist(pos_snap, goal_snap)
        if real_dist >= self.goal_threshold:
            return False

        # Condición 2: confirmación desde odom_callback
        # (self.goal_distance se actualizó con el goal actual)
        if self.goal_distance >= self.goal_threshold:
            return False

        # Condición 3: mínimo de pasos desde el reset para evitar
        # falsos positivos inmediatos (robot spawna cerca del nuevo goal)
        if self.episode_step < self.MIN_STEPS_BEFORE_GOAL:
            self.get_logger().warn(
                f"Goal distance={real_dist:.3f}m < threshold pero "
                f"episode_step={self.episode_step} < {self.MIN_STEPS_BEFORE_GOAL} "
                f"(posible falso positivo post-reset, ignorado)"
            )
            return False

        return True

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def distance_to_goal(self) -> float:
        return self._dist(self.position, self.goal_position)

    # -------------------------------------------------------
    # Velocity & State
    # -------------------------------------------------------
    def send_velocity(self, linear: float, angular: float):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.twist.linear.x  = linear
        msg.twist.linear.y  = 0.0
        msg.twist.linear.z  = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def get_state(self) -> np.ndarray:
        return None

    # -------------------------------------------------------
    # Reset diferenciado
    #
    # reset_full():      colision → resetea robot Y genera nuevo objetivo
    # reset_goal_only(): exito/timeout → solo genera nuevo objetivo,
    #                    robot mantiene su posicion actual
    # -------------------------------------------------------
    def reset_full(self) -> np.ndarray:
        """Reset completo: usado SOLO en colision."""
        self.send_velocity(0.0, 0.0)
        # reset_robot_pose() ya incluye spins internos para actualizar /odom
        self.reset_robot_pose()
        # Pausa adicional para que Gazebo estabilice la física tras el teleport
        time.sleep(0.3)

        if self.entity_spawned:
            self.delete_entity()
        # Generar goal con distancia minima valida respecto al origen (0,0).
        # Usar (0.0, 0.0) explícitamente, NO self.position, porque aunque
        # reset_robot_pose() hace spins, /odom puede tener lag residual.
        self.generate_goal_pose(robot_pos=(0.0, 0.0))
        self.spawn_entity()

        # Múltiples spins para garantizar que position se actualice
        # con la posición real del robot ANTES del primer is_goal_reached()
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.last_distance   = self.distance_to_goal()
        self.previous_action = None
        self.action_history  = []
        self.episode_step    = 0   # resetear contador de pasos
        self.get_logger().info(
            f'reset_full complete: robot=({self.position[0]:.3f},{self.position[1]:.3f}) '
            f'goal=({self.goal_position[0]:.3f},{self.goal_position[1]:.3f}) '
            f'dist={self.last_distance:.3f}m'
        )
        return self.get_state()

    def reset_goal_only(self) -> np.ndarray:
        """Solo nuevo objetivo: usado en exito o timeout."""
        self.send_velocity(0.0, 0.0)

        if self.entity_spawned:
            self.delete_entity()
        # Generar goal con distancia minima valida respecto a la posicion ACTUAL
        self.generate_goal_pose(robot_pos=self.position)
        self.spawn_entity()

        # Múltiples spins para que position y goal_distance se sincronicen
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.last_distance   = self.distance_to_goal()
        self.previous_action = None
        self.action_history  = []
        self.episode_step    = 0   # resetear contador de pasos
        return self.get_state()

    # Mantener reset() por compatibilidad (llama reset_full)
    def reset(self, random_goal: bool = True) -> np.ndarray:
        return self.reset_full()

    # -------------------------------------------------------
    # Gazebo entity management
    # -------------------------------------------------------
    def _load_entity_sdf(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            import os as _os
            package_share = get_package_share_directory('turtlebot3_gazebo')
            self.entity_model_path = _os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            self.get_logger().info(f'goal_box model path: {self.entity_model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to resolve goal_box path: {e}')
            self.entity_model_path = None

    def spawn_entity(self):
        """
        Spawn del goal marker.
        Solo intenta delete si entity_spawned=True para evitar el error
        de Gazebo "Entity not found" cuando no existe marker previo.
        entity_spawned se mantiene consistente via _force_delete_entity/finally.
        """
        if self.entity_model_path is None:
            self.get_logger().error('Cannot spawn goal_box: model path not resolved')
            return

        # Solo borrar si el flag indica que hay uno activo en Gazebo
        if self.entity_spawned:
            self._force_delete_entity()
            time.sleep(0.15)   # dar tiempo a Gazebo para procesar el delete

        x, y = self.goal_position
        req = (
            f'sdf_filename: "{self.entity_model_path}", '
            f'name: "{self.entity_name}", '
            f'pose: {{ position: {{ x: {x}, y: {y}, z: 0.0 }} }}'
        )
        cmd = ['gz', 'service', '-s', '/world/dqn/create',
               '--reqtype', 'gz.msgs.EntityFactory',
               '--reptype', 'gz.msgs.Boolean',
               '--timeout', '1000', '--req', req]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.get_logger().info(f'Goal marker spawned at ({x}, {y})')
                self.entity_spawned = True
            else:
                self.get_logger().warn(f'spawn_entity non-zero: {result.stderr}')
                self.entity_spawned = False
        except Exception as e:
            self.get_logger().error(f'spawn_entity error: {e}')
            self.entity_spawned = False

    def delete_entity(self):
        """Delete con actualizacion de flag garantizada."""
        self._force_delete_entity()

    def _force_delete_entity(self):
        """
        Intenta eliminar el goal_box de Gazebo.
        Marca entity_spawned=False SIEMPRE, porque si Gazebo
        retorna error probablemente ya no existia el marker.
        """
        cmd = ['gz', 'service', '-s', '/world/dqn/remove',
               '--reqtype', 'gz.msgs.Entity',
               '--reptype', 'gz.msgs.Boolean',
               '--timeout', '1000',
               '--req', f'name: "{self.entity_name}", type: 2']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.get_logger().info('Goal marker deleted')
            else:
                self.get_logger().debug(
                    f'delete_entity non-zero (probable ya no existia): {result.stderr}'
                )
        except Exception as e:
            self.get_logger().error(f'delete_entity error: {e}')
        finally:
            # Siempre marcar como no spawneado para evitar doble spawn
            self.entity_spawned = False

    def publish_goal_pose(self):
        """
        Publica la goal_position actual en /goal_pose (frame: odom).
        Permite verificar externamente que lo que el código usa como goal
        coincide con el marker visual — única fuente de verdad.
        """
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'   # mismo frame que /odom del robot
        msg.pose.position.x = float(self.goal_position[0])
        msg.pose.position.y = float(self.goal_position[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pose_pub.publish(msg)

    def generate_goal_pose(self, robot_pos: Tuple[float, float] = (0.0, 0.0)):
        """
        Genera un nuevo objetivo garantizando distancia minima al robot.
        Reintenta hasta MAX_GOAL_RETRIES veces antes de aceptar cualquier posicion.
        """
        if self.stage != 4:
            for _ in range(self.MAX_GOAL_RETRIES):
                x = random.randrange(-21, 21) / 10
                y = random.randrange(-21, 21) / 10
                dist = math.sqrt((x - robot_pos[0]) ** 2 + (y - robot_pos[1]) ** 2)
                if dist >= self.MIN_GOAL_DISTANCE:
                    break
            # Si agota intentos, forzar posicion minima valida
            else:
                angle = random.uniform(0, 2 * math.pi)
                x = robot_pos[0] + self.MIN_GOAL_DISTANCE * math.cos(angle)
                y = robot_pos[1] + self.MIN_GOAL_DISTANCE * math.sin(angle)
                x = max(-2.0, min(2.0, x))
                y = max(-2.0, min(2.0, y))
        else:
            goal_pose_list = [
                [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 1.5], [0.5, 2.0], [-1.5, 2.1],
                [-2.0, 0.5], [-2.0, -0.5], [-1.5, -2.0], [-0.5, -1.0], [2.0, -0.5], [-1.0, -1.0]
            ]
            # Filtrar posiciones demasiado cercanas
            valid = [p for p in goal_pose_list
                     if math.sqrt((p[0] - robot_pos[0])**2 + (p[1] - robot_pos[1])**2)
                     >= self.MIN_GOAL_DISTANCE]
            chosen = random.choice(valid if valid else goal_pose_list)
            x, y   = chosen

        self.goal_position = (x, y)
        self.get_logger().info(
            f'New goal: ({x:.2f}, {y:.2f}) | '
            f'dist to robot: {math.sqrt((x-robot_pos[0])**2+(y-robot_pos[1])**2):.2f}m'
        )
        # Publicar inmediatamente para que /goal_pose refleje el nuevo objetivo
        self.publish_goal_pose()

    def reset_robot_pose(self):
        """
        Resetea el robot al origen usando delete + respawn.
        Método copiado de dqn_gazebo.py (reset_burger) que es el mecanismo
        confirmado como funcional para este setup de Gazebo.

        NOTA: el nombre de instancia en Gazebo es 'burger' pero el SDF
        está en models/turtlebot3_burger/model.sdf — ambos deben coincidir
        exactamente con los valores usados en dqn_gazebo.py.
        """
        self.send_velocity(0.0, 0.0)

        # 1. Borrar el robot actual de Gazebo
        cmd_delete = [
            'gz', 'service',
            '-s', '/world/dqn/remove',
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', f'name: "{self.robot_name}", type: 2'
        ]
        try:
            subprocess.run(cmd_delete, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            self.get_logger().info(f'Delete robot "{self.robot_name}" OK')
        except subprocess.CalledProcessError:
            self.get_logger().warn(f'Delete robot failed (puede que ya no existiera)')

        time.sleep(0.2)

        # 2. Resolver el path del SDF del burger
        try:
            from ament_index_python.packages import get_package_share_directory as _gpsd
            import os as _os
            _pkg = _gpsd('turtlebot3_gazebo')
            model_path = _os.path.join(_pkg, 'models', 'turtlebot3_burger', 'model.sdf')
        except Exception as e:
            self.get_logger().warn(f'ament_index falló ({e}), usando path absoluto')
            model_path = ('/opt/ros/humble/share/turtlebot3_gazebo/models/'
                          'turtlebot3_burger/model.sdf')

        # 3. Re-spawnear el robot en el origen (0, 0, 0)
        req_spawn = (
            f'sdf_filename: "{model_path}", '
            f'name: "{self.robot_name}", '
            f'pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.0 }} }}'
        )
        cmd_spawn = [
            'gz', 'service',
            '-s', '/world/dqn/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_spawn
        ]
        try:
            subprocess.run(cmd_spawn, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            self.get_logger().info(f'Spawn robot "{self.robot_name}" en (0,0) OK')
        except subprocess.CalledProcessError:
            self.get_logger().error('Spawn robot falló')

        # 4. Esperar a que Gazebo procese el spawn y /odom se estabilice
        time.sleep(0.2)
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(
            f'Post-respawn position: ({self.position[0]:.3f}, {self.position[1]:.3f})'
        )