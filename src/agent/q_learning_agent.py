"""Implementação tabular de Q-learning para lane following."""

import os
import numpy as np
from typing import Sequence

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path)

    def decay_epsilon(self, min_epsilon: float, decay_rate: float) -> None:
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
