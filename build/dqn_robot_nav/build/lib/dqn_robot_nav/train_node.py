#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import numpy as np
from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
from dqn_robot_nav.state_processor import StateProcessor
import matplotlib.pyplot as plt
from datetime import datetime
import os


class DQNTrainingNode(Node):
    """Nodo principal de entrenamiento DQN para navegacion TurtleBot3"""

    def __init__(self):
        super().__init__('dqn_training_node')

        self.n_episodes            = 220
        self.max_steps_per_episode = 1500   # limite estricto de pasos por episodio

        self.state_size  = 12   # 10 bins LiDAR + distancia + angulo al objetivo
        self.action_size = 7    # 7 acciones (sin movimiento hacia atras)

        # Componentes principales
        self.env             = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=10)

        self.agent = DQNAgent(
            state_size          = self.state_size,
            action_size         = self.action_size,
            learning_rate       = 0.0003,
            gamma               = 0.99,
            epsilon             = 1.0,
            epsilon_min         = 0.05,
            epsilon_decay_steps = 6000,   # decay exponencial
            memory_size         = 20000,
            batch_size          = 64,
            target_update_freq  = 200
        )

        # Metricas
        self.episode_rewards       = []
        self.episode_steps         = []
        self.episode_distances     = []
        self.episode_min_distances = []
        self.success_count   = 0
        self.collision_count = 0
        self.timeout_count   = 0

        # Historial por episodio para grafica de outcomes correcta
        self.outcome_history = []   # lista de 'success' | 'collision' | 'timeout'

        # Directorio de resultados
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.log_file = os.path.join(self.results_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write("Episode,Steps,Reward,Outcome,Final_Distance,Min_Distance,Epsilon\n")

    def get_processed_state(self) -> np.ndarray:
        return self.state_processor.get_state(
            self.env.scan_data,
            self.env.position,
            self.env.goal_position,
            self.env.yaw
        )

    @staticmethod
    def _frozen_dist(pos: tuple, goal: tuple) -> float:
        """
        Calcula distancia entre pos y goal usando solo las coordenadas
        pasadas como argumento. No accede a ninguna variable del env,
        eliminando cualquier posibilidad de race condition.
        """
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def train(self):
        """
        Bucle principal de entrenamiento.

        Logica de reset diferenciada:
          - Colision   → reset_full()  : robot + objetivo se reinician
          - Exito      → reset_goal_only(): solo nuevo objetivo, robot donde esta
          - Timeout    → reset_goal_only(): solo nuevo objetivo, robot donde esta

        El step_count del agente es el unico que controla el epsilon decay.
        El limite de pasos es estrictamente self.max_steps_per_episode.
        """
        self.get_logger().info("Iniciando entrenamiento DQN...")
        self.get_logger().info(f"  Episodios: {self.n_episodes}")
        self.get_logger().info(f"  Pasos max por episodio: {self.max_steps_per_episode}")

        # Reset inicial completo
        self.env.reset_full()
        rclpy.spin_once(self.env, timeout_sec=0.5)

        for episode in range(self.n_episodes):

            # -----------------------------------------------------------
            # CONGELAR goal_position AL INICIO DEL EPISODIO.
            # Spin extra para asegurar que goal_position ya refleja el
            # estado post-reset antes de congelar.
            # goal_frozen NUNCA cambia durante el episodio.
            # -----------------------------------------------------------
            rclpy.spin_once(self.env, timeout_sec=0.1)

            # Log de diagnóstico al inicio del episodio
            self.get_logger().info(
                f"Ep {episode}: goal=({self.env.goal_position[0]:.3f},{self.env.goal_position[1]:.3f}) "
                f"robot=({self.env.position[0]:.3f},{self.env.position[1]:.3f}) "
                f"dist_initial={self.env.goal_distance:.3f}m"
            )

            state             = self.get_processed_state()
            episode_reward    = 0.0
            min_dist_to_goal  = self.env.goal_distance
            outcome           = 'timeout'

            # -------------------------------------------------------
            # Bucle de pasos — ESTRICTAMENTE limitado a max_steps
            # -------------------------------------------------------
            for step in range(self.max_steps_per_episode):

                action             = self.agent.act(state, training=True)
                _, reward, done    = self.env.step(action)
                next_state         = self.get_processed_state()

                # -------------------------------------------------------
                # GOAL CHECK — réplica exacta de dqn_environment.py:
                #   if self.goal_distance < 0.20  (calculado en odom_callback)
                #
                # self.env.goal_distance se actualiza en odom_callback usando
                # self.env.goal_position, que es la misma variable que controla
                # el marker visual en Gazebo. Fuente única de verdad — sin
                # divergencia posible entre lo que ve el código y lo visual.
                # -------------------------------------------------------
                if not done:   # solo verificar si no hubo colisión
                    if self.env.goal_distance < self.env.goal_threshold:
                        reward += 200.0
                        done    = True
                        outcome = 'goal'
                        self.env._last_done_reason = 'goal'
                        self.get_logger().info(
                            f"GOAL REACHED ep={episode} step={step} "
                            f"goal_distance={self.env.goal_distance:.3f}m "
                            f"robot=({self.env.position[0]:.2f},{self.env.position[1]:.2f}) "
                            f"goal=({self.env.goal_position[0]:.2f},{self.env.goal_position[1]:.2f})"
                        )

                self.agent.remember(state, action, reward, next_state, done)

                if len(self.agent.memory) >= self.agent.batch_size:
                    self.agent.replay()

                episode_reward += reward
                state           = next_state

                if self.env.goal_distance < min_dist_to_goal:
                    min_dist_to_goal = self.env.goal_distance

                if done:
                    if outcome == 'timeout':   # done por colision
                        outcome = self.env._last_done_reason
                    break

                rclpy.spin_once(self.env, timeout_sec=0.01)

            # -------------------------------------------------------
            # Reset segun tipo de terminacion (UNA SOLA VEZ aqui)
            # -------------------------------------------------------
            # final_distance desde odom_callback — misma fuente que el goal check
            final_distance = self.env.goal_distance

            if outcome == 'collision':
                self.collision_count += 1
                self.env.reset_full()
                # reset_full() ya hace spins internos, pero hacemos spins extra
                # en el train_node para absorber cualquier lag residual de Gazebo
                # antes de congelar goal_frozen en la próxima iteración del episodio.
                for _ in range(5):
                    rclpy.spin_once(self.env, timeout_sec=0.1)
                self.get_logger().info(
                    f'Post-collision reset: robot=({self.env.position[0]:.3f},'
                    f'{self.env.position[1]:.3f})'
                )

            elif outcome == 'goal':
                self.success_count += 1
                self.env.reset_goal_only()
                for _ in range(3):
                    rclpy.spin_once(self.env, timeout_sec=0.1)

            else:   # timeout
                self.timeout_count += 1
                self.env.reset_goal_only()
                for _ in range(3):
                    rclpy.spin_once(self.env, timeout_sec=0.1)

            # -------------------------------------------------------
            # Registro de metricas
            # -------------------------------------------------------
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)
            self.episode_distances.append(final_distance)
            self.episode_min_distances.append(min_dist_to_goal)
            self.outcome_history.append(outcome)

            with open(self.log_file, 'a') as f:
                f.write(
                    f"{episode},{step+1},{episode_reward:.2f},{outcome},"
                    f"{final_distance:.3f},{min_dist_to_goal:.3f},"
                    f"{self.agent.epsilon:.4f}\n"
                )

            if episode % 10 == 0:
                avg_reward     = np.mean(self.episode_rewards[-10:])
                success_rate   = self.success_count   / (episode + 1) * 100
                collision_rate = self.collision_count / (episode + 1) * 100
                timeout_rate   = self.timeout_count   / (episode + 1) * 100
                avg_dist       = np.mean(self.episode_distances[-10:])

                self.get_logger().info(
                    f"Ep {episode:4d}/{self.n_episodes} | "
                    f"Steps: {step+1:3d} | "
                    f"Reward: {episode_reward:7.1f} | "
                    f"AvgR(10): {avg_reward:7.1f} | "
                    f"ε: {self.agent.epsilon:.3f}"
                )
                self.get_logger().info(
                    f"  ✓ {success_rate:5.1f}%  "
                    f"✗ {collision_rate:5.1f}%  "
                    f"⏱ {timeout_rate:5.1f}%  "
                    f"AvgDist: {avg_dist:.2f}m"
                )

            # Guardar modelo periodicamente
            if (episode < 100 and episode % 25 == 0 and episode > 0) or \
               (episode >= 100 and episode % 50 == 0):
                path = os.path.join(self.results_dir, f"model_ep{episode}.pkl")
                self.agent.save(path)

        # Guardado final
        self.agent.save(os.path.join(self.results_dir, "model_final.pkl"))
        self.plot_results()

    def plot_results(self):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Recompensas por episodio
        axes[0, 0].plot(self.episode_rewards, alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)

        # Media movil
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards,
                                     np.ones(window) / window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Reward (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True)

        # Pasos por episodio
        axes[1, 0].plot(self.episode_steps, alpha=0.7)
        axes[1, 0].axhline(y=self.max_steps_per_episode, color='r',
                           linestyle='--', label='Max Steps')
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Outcomes acumulados correctos (calculados por episodio)
        success_r   = []
        collision_r = []
        timeout_r   = []
        for i, o in enumerate(self.outcome_history):
            n = i + 1
            success_r.append(  self.outcome_history[:n].count('goal')      / n * 100)
            collision_r.append(self.outcome_history[:n].count('collision') / n * 100)
            timeout_r.append(  self.outcome_history[:n].count('timeout')   / n * 100)

        axes[1, 1].plot(success_r,   label='Success',   color='green')
        axes[1, 1].plot(collision_r, label='Collision',  color='red')
        axes[1, 1].plot(timeout_r,   label='Timeout',    color='orange')
        axes[1, 1].set_title('Episode Outcomes (cumulative %)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Distancia final al objetivo
        axes[2, 0].plot(self.episode_distances, alpha=0.7)
        axes[2, 0].axhline(y=self.env.goal_threshold, color='g',
                           linestyle='--', label='Goal Threshold')
        axes[2, 0].set_title('Final Distance to Goal')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Distance (m)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Distancia minima alcanzada
        axes[2, 1].plot(self.episode_min_distances, alpha=0.7)
        axes[2, 1].axhline(y=self.env.goal_threshold, color='g',
                           linestyle='--', label='Goal Threshold')
        axes[2, 1].set_title('Min Distance Achieved per Episode')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Min Distance (m)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_results.png'), dpi=150)
        self.get_logger().info(f"Graficas guardadas en {self.results_dir}")

        # Estadisticas finales
        stats_file = os.path.join(self.results_dir, 'final_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("=== TRAINING STATISTICS ===\n\n")
            f.write(f"Total Episodes:          {self.n_episodes}\n")
            f.write(f"Max Steps per Episode:   {self.max_steps_per_episode}\n\n")
            f.write(f"Success Rate:   {self.success_count   / self.n_episodes * 100:.2f}%\n")
            f.write(f"Collision Rate: {self.collision_count / self.n_episodes * 100:.2f}%\n")
            f.write(f"Timeout Rate:   {self.timeout_count   / self.n_episodes * 100:.2f}%\n\n")
            f.write(f"Average Reward:         {np.mean(self.episode_rewards):.2f}\n")
            f.write(f"Average Steps:          {np.mean(self.episode_steps):.2f}\n")
            f.write(f"Average Final Distance: {np.mean(self.episode_distances):.3f}m\n")
            f.write(f"Average Min Distance:   {np.mean(self.episode_min_distances):.3f}m\n\n")
            f.write(f"Final Epsilon:          {self.agent.epsilon:.4f}\n")
            f.write(f"Total Replay Steps:     {self.agent.step_count}\n")


def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Entrenamiento interrumpido por el usuario")
    finally:
        trainer.env.send_velocity(0.0, 0.0)
        trainer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()