import numpy as np
from typing import Tuple
import math


class StateProcessor:
    """
    Procesa datos LiDAR y odometria en el vector de estado para el DQN.
    Estado: [10 bins LiDAR normalizados, goal_distance_norm, goal_angle_norm]
    Total: 12 dimensiones
    """

    def __init__(self, n_lidar_bins: int = 10):
        self.n_lidar_bins = n_lidar_bins
        self.max_lidar_range = 3.5  # TurtleBot3 LiDAR max range

    def process_lidar(self, scan_data: list) -> np.ndarray:
        """
        Discretiza el scan LiDAR de 360 puntos en n_lidar_bins sectores.
        Usa el percentil 10 de cada sector (robusto ante ruido, casi-minimo).

        Returns: array (n_lidar_bins,) normalizado en [0, 1]
        """
        scan_array = np.array(scan_data, dtype=float)

        # Limpiar valores invalidos
        scan_array[np.isinf(scan_array)] = self.max_lidar_range
        scan_array[np.isnan(scan_array)] = self.max_lidar_range
        scan_array[scan_array <= 0.0]    = self.max_lidar_range
        scan_array = np.clip(scan_array, 0.0, self.max_lidar_range)

        points_per_bin = len(scan_array) // self.n_lidar_bins
        binned_scan = []

        for i in range(self.n_lidar_bins):
            start_idx = i * points_per_bin
            end_idx = (
                (i + 1) * points_per_bin
                if i < self.n_lidar_bins - 1
                else len(scan_array)
            )
            bin_values = scan_array[start_idx:end_idx]
            # Percentil 10: casi-minimo pero robusto ante ruido espurio
            binned_scan.append(np.percentile(bin_values, 10))

        return np.array(binned_scan) / self.max_lidar_range

    def compute_goal_info(self,
                          current_pos: Tuple[float, float],
                          goal_pos: Tuple[float, float],
                          current_yaw: float) -> np.ndarray:
        """
        Calcula distancia y angulo relativo al objetivo.
        Usa la misma logica que dqn_environment.py de ROBOTIS.

        Returns: [distance_norm, angle_norm]
            distance_norm: [0, 1]  (normalizado a 10m max)
            angle_norm:    [-1, 1] (normalizado a pi)
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]

        distance = math.sqrt(dx ** 2 + dy ** 2)

        path_theta = math.atan2(dy, dx)
        relative_angle = path_theta - current_yaw

        # Normalizar a [-pi, pi]
        if relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        elif relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        distance_norm = float(np.clip(distance / 10.0, 0.0, 1.0))
        angle_norm    = float(relative_angle / math.pi)   # [-1, 1]

        return np.array([distance_norm, angle_norm])

    def get_state(self,
                  scan_data: list,
                  current_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  current_yaw: float) -> np.ndarray:
        """
        Combina LiDAR y goal info en el vector de estado completo.

        Returns: array (n_lidar_bins + 2,) = (12,)
        """
        if scan_data is None or len(scan_data) == 0:
            lidar_state = np.ones(self.n_lidar_bins)  # estado conservador
        else:
            lidar_state = self.process_lidar(scan_data)

        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)

        state = np.concatenate([lidar_state, goal_state])

        # Sanity check
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print("Warning: Invalid state detected! Using safe default.")
            state = np.nan_to_num(state, nan=0.5, posinf=1.0, neginf=0.0)

        return state