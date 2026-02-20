#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
from dqn_robot_nav.state_processor import StateProcessor
import numpy as np


class DQNTestNode(Node):
    """Test del agente DQN entrenado — política greedy pura (epsilon=0)."""

    def __init__(self):
        super().__init__('dqn_test_node')

        # -------------------------------------------------------
        # Parámetros ROS — pasar con:
        #   ros2 run dqn_robot_nav test_node --ros-args \
        #       -p model_path:=/ruta/al/model_final.pkl \
        #       -p n_episodes:=10
        # -------------------------------------------------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('n_episodes', 10)
        self.declare_parameter('max_steps',  1500)

        model_path      = self.get_parameter('model_path').get_parameter_value().string_value
        self.n_episodes = self.get_parameter('n_episodes').get_parameter_value().integer_value
        self.max_steps  = self.get_parameter('max_steps').get_parameter_value().integer_value

        # -------------------------------------------------------
        # Validar que se proporcionó un modelo
        # -------------------------------------------------------
        if not model_path:
            self.get_logger().error(
                'Falta el parámetro model_path.\n'
                'Uso correcto:\n'
                '  ros2 run dqn_robot_nav test_node --ros-args '
                '-p model_path:=/ruta/al/model_final.pkl'
            )
            raise SystemExit(1)

        import os
        if not os.path.isfile(model_path):
            self.get_logger().error(f'Archivo no encontrado: {model_path}')
            raise SystemExit(1)

        # -------------------------------------------------------
        # Misma configuración que en train_node.py
        # -------------------------------------------------------
        self.state_size  = 12   # 10 bins LiDAR + distancia + ángulo
        self.action_size = 7    # debe coincidir con train_node.py

        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            memory_size=1,
            batch_size=1
        )
        self.agent.load(model_path)
        self.agent.epsilon = 0.0   # greedy puro, sin exploración

        self.env             = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=10)

        self.get_logger().info(f'Modelo cargado: {model_path}')
        self.get_logger().info(
            f'Configuración: state={self.state_size}, actions={self.action_size}, '
            f'episodios={self.n_episodes}, max_steps={self.max_steps}'
        )

    def get_processed_state(self):
        return self.state_processor.get_state(
            self.env.scan_data,
            self.env.position,
            self.env.goal_position,
            self.env.yaw
        )

    def test(self):
        """Ejecuta n_episodes episodios de evaluación."""
        successes   = 0
        collisions  = 0
        timeouts    = 0
        total_rewards = []

        # Reset inicial
        self.env.reset_full()
        for _ in range(5):
            rclpy.spin_once(self.env, timeout_sec=0.1)

        for episode in range(self.n_episodes):

            rclpy.spin_once(self.env, timeout_sec=0.1)
            self.get_logger().info(
                f'[Ep {episode+1}/{self.n_episodes}] '
                f'goal=({self.env.goal_position[0]:.2f},{self.env.goal_position[1]:.2f}) '
                f'robot=({self.env.position[0]:.2f},{self.env.position[1]:.2f}) '
                f'dist={self.env.goal_distance:.2f}m'
            )

            state          = self.get_processed_state()
            episode_reward = 0.0
            outcome        = 'timeout'

            for step in range(self.max_steps):

                action          = self.agent.act(state, training=False)
                _, reward, done = self.env.step(action)
                next_state      = self.get_processed_state()

                # Goal check — idéntico a train_node
                if not done:
                    if self.env.goal_distance < self.env.goal_threshold:
                        reward += 200.0
                        done    = True
                        outcome = 'goal'
                        self.env._last_done_reason = 'goal'

                episode_reward += reward
                state           = next_state

                if done:
                    if outcome == 'timeout' and self.env._last_done_reason == 'collision':
                        outcome = 'collision'
                    break

                rclpy.spin_once(self.env, timeout_sec=0.01)

            # -------------------------------------------------------
            # Log del resultado del episodio
            # -------------------------------------------------------
            icon = {'goal': '✓', 'collision': '✗', 'timeout': '⏱'}[outcome]
            self.get_logger().info(
                f'  {icon} Ep {episode+1}: {outcome.upper()} | '
                f'steps={step+1} | reward={episode_reward:.1f} | '
                f'dist_final={self.env.goal_distance:.2f}m'
            )

            # -------------------------------------------------------
            # Reset diferenciado — igual que train_node
            # reset_full SOLO en colisión; timeout/goal → reset_goal_only
            # -------------------------------------------------------
            if outcome == 'collision':
                collisions += 1
                self.env.reset_full()
                for _ in range(5):
                    rclpy.spin_once(self.env, timeout_sec=0.1)
            elif outcome == 'goal':
                successes += 1
                self.env.reset_goal_only()
                for _ in range(3):
                    rclpy.spin_once(self.env, timeout_sec=0.1)
            else:   # timeout
                timeouts += 1
                self.env.reset_goal_only()
                for _ in range(3):
                    rclpy.spin_once(self.env, timeout_sec=0.1)

            total_rewards.append(episode_reward)

        # -------------------------------------------------------
        # Resumen final
        # -------------------------------------------------------
        n = self.n_episodes
        self.get_logger().info('=' * 52)
        self.get_logger().info('=== RESULTADOS DE EVALUACIÓN ====================')
        self.get_logger().info(f'  Episodios  : {n}')
        self.get_logger().info(f'  ✓ Éxitos   : {successes}  ({successes/n*100:.1f}%)')
        self.get_logger().info(f'  ✗ Colisiones: {collisions} ({collisions/n*100:.1f}%)')
        self.get_logger().info(f'  ⏱ Timeouts : {timeouts}  ({timeouts/n*100:.1f}%)')
        self.get_logger().info(f'  Reward medio: {np.mean(total_rewards):.2f}')
        self.get_logger().info(f'  Reward std  : {np.std(total_rewards):.2f}')
        self.get_logger().info('=' * 52)

        self.env.send_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = DQNTestNode()
    try:
        node.test()
    except KeyboardInterrupt:
        node.get_logger().info('Test interrumpido')
    finally:
        node.env.send_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()