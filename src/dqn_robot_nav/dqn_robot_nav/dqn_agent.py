import numpy as np
from sklearn.neural_network import MLPRegressor
from collections import deque
import random
import pickle
import math


class DQNAgent:
    """
    Deep Q-Network agent usando sklearn MLPRegressor.
    - Double DQN para reducir sobreestimacion de Q-values
    - Epsilon decay EXPONENCIAL (de dqn_agent.py de ROBOTIS)
    - Experience replay buffer
    - Target network con actualizacion periodica
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay_steps: int = 6000,
                 memory_size: int = 20000,
                 batch_size: int = 64,
                 target_update_freq: int = 200):
        """
        Args:
            state_size:           Dimension del espacio de estado
            action_size:          Numero de acciones discretas
            learning_rate:        Tasa de aprendizaje
            gamma:                Factor de descuento
            epsilon:              Exploracion inicial
            epsilon_min:          Exploracion minima
            epsilon_decay_steps:  Pasos hasta llegar a epsilon_min (decay exponencial)
                                  Formula: epsilon = epsilon_min + (1-epsilon_min) * exp(-step/decay_steps)
                                  Con 6000 pasos, a los ~4000 steps epsilon ~ 0.10
            memory_size:          Tamano del replay buffer
            batch_size:           Batch para entrenamiento
            target_update_freq:   Steps entre actualizaciones de target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps  # para decay exponencial
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0       # contador acumulado de llamadas a replay()

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Q-network: arquitectura 256-256-128 (misma que ROBOTIS pero con sklearn)
        self.q_network = MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42,
            alpha=0.0001,   # regularizacion L2 para evitar overfitting
            batch_size='auto',
            shuffle=True
        )

        # Target network (misma arquitectura)
        self.target_network = MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42,
            alpha=0.0001,
            batch_size='auto',
            shuffle=True
        )

        # Inicializar redes con datos dummy para que partial_fit funcione
        dummy_X = np.random.randn(max(batch_size, 10), state_size)
        dummy_y = np.random.randn(max(batch_size, 10), action_size)
        self.q_network.fit(dummy_X, dummy_y)
        self.target_network.fit(dummy_X, dummy_y)

        self.loss_history = []

    def remember(self, state, action, reward, next_state, done):
        """Almacenar experiencia en el replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleccionar accion usando politica epsilon-greedy con decay EXPONENCIAL.
        Formula de ROBOTIS:
          epsilon = epsilon_min + (1 - epsilon_min) * exp(-step_count / epsilon_decay_steps)
        """
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print("Warning: Invalid state in act(), using random action")
            return random.randrange(self.action_size)

        # Actualizacion exponencial del epsilon (en cada llamada a act durante training)
        if training:
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_count / self.epsilon_decay_steps
            )

        # Exploracion
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Explotacion
        try:
            state_reshaped = state.reshape(1, -1)
            q_values = self.q_network.predict(state_reshaped)[0]
            if np.any(np.isnan(q_values)) or np.any(np.isinf(q_values)):
                print("Warning: Invalid Q-values, using random action")
                return random.randrange(self.action_size)
            return int(np.argmax(q_values))
        except Exception as e:
            print(f"Error in act(): {e}, using random action")
            return random.randrange(self.action_size)

    def replay(self) -> float:
        """
        Entrenar la Q-network con un batch del replay buffer.
        Implementa Double DQN para reducir sobreestimacion.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.array([t[0] for t in minibatch])
        actions     = np.array([t[1] for t in minibatch])
        rewards     = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones       = np.array([t[4] for t in minibatch])

        # Clip rewards para estabilidad
        rewards = np.clip(rewards, -300, 300)

        try:
            current_q_values   = self.q_network.predict(states)
            next_q_main        = self.q_network.predict(next_states)    # para seleccion (Double DQN)
            next_q_target      = self.target_network.predict(next_states)  # para evaluacion

            target_q_values = current_q_values.copy()

            for i in range(self.batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    # Double DQN: seleccionar accion con main, evaluar con target
                    best_action = int(np.argmax(next_q_main[i]))
                    target_q_values[i][actions[i]] = (
                        rewards[i] + self.gamma * next_q_target[i][best_action]
                    )

            target_q_values = np.clip(target_q_values, -500, 500)

            self.q_network.partial_fit(states, target_q_values)

            # Actualizar step_count y target network periodicamente
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()

            # Calcular y registrar loss (MSE)
            loss = float(np.mean((target_q_values - current_q_values) ** 2))
            self.loss_history.append(loss)
            if len(self.loss_history) > 1000:
                self.loss_history.pop(0)

            return loss

        except Exception as e:
            print(f"Error in replay(): {e}")
            return 0.0

    def update_target_network(self):
        """Copiar pesos de Q-network a target network."""
        self.target_network = pickle.loads(pickle.dumps(self.q_network))

    def save(self, filepath: str):
        model_data = {
            'q_network':      self.q_network,
            'target_network': self.target_network,
            'epsilon':        self.epsilon,
            'step_count':     self.step_count,
            'loss_history':   self.loss_history,
            'state_size':     self.state_size,
            'action_size':    self.action_size,
            'gamma':          self.gamma,
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.q_network      = model_data['q_network']
            self.target_network = model_data['target_network']
            self.epsilon        = model_data['epsilon']
            self.step_count     = model_data['step_count']
            if 'loss_history' in model_data:
                self.loss_history = model_data['loss_history']
            print(f"Model loaded from {filepath}")
            print(f"  epsilon={self.epsilon:.4f}, step_count={self.step_count}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_average_loss(self, window: int = 100) -> float:
        if not self.loss_history:
            return 0.0
        return float(np.mean(self.loss_history[-window:]))