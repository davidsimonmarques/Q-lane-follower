"""Implementação tabular de Q-learning para lane following."""

import os
import logging
import numpy as np
from typing import Sequence

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.config = config
        
        # Inicializar Q-table com controle de carregamento
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self._initialize_qtable()

    def _initialize_qtable(self) -> None:
        """Inicializa a Q-table: carrega pré-treinada ou cria nova."""
        q_table_path = self.config.get("q_table_path", "assets/q_table.npy")
        load_pretrained = self.config.get("load_pretrained", True)
        
        logger = logging.getLogger("lane_follower")
        
        # Se load_pretrained=True e arquivo existe, carregar
        if load_pretrained and os.path.exists(q_table_path):
            try:
                self.load(q_table_path)
                logger.info(f"✅ Q-table carregada de: {q_table_path}")
                logger.info(f"   Continuando treinamento a partir da tabela existente")
            except Exception as e:
                logger.warning(f"⚠️  Erro ao carregar Q-table: {e}")
                logger.info("   Criando nova Q-table do zero")
                self.q_table = np.zeros((self.state_size, self.action_size), dtype=np.float32)
        else:
            # Se load_pretrained=False ou arquivo não existe, criar nova
            if not load_pretrained:
                logger.info("📊 load_pretrained=False: começando treinamento do zero")
            else:
                logger.info(f"📊 Nenhuma Q-table encontrada em {q_table_path}: criando nova")
            self.q_table = np.zeros((self.state_size, self.action_size), dtype=np.float32)

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
