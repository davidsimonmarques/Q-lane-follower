"""Cálculo de recompensa específico para lane following."""

from typing import Dict
import numpy as np

class RewardFunction:
    def __init__(self, config: Dict):
        self.lane_center_penalty = config.get("lane_center_penalty", 1.0)
        self.heading_penalty = config.get("heading_penalty", 0.5)
        self.offroad_penalty = config.get("offroad_penalty", 5.0)
        self.speed_reward = config.get("speed_reward", 0.1)
        self.speed_penalty_threshold = config.get("speed_penalty_threshold", 1.0)
        self.speed_penalty_weight = config.get("speed_penalty_weight", 0.1)

    def compute(self, observation: Dict, action: int, done: bool) -> float:
        # 1. Configurações de limites (pode colocar no config)
        MAX_OFFSET = 2.0  # Distância máxima da linha antes de ser considerado fora
        
        lane_offset = abs(observation.get("lane_offset", 0.0))
        is_offroad = observation.get("offroad", False) or (lane_offset > MAX_OFFSET)

        # 2. Se bateu ou saiu da pista: Punição Crítica e Fim
        if is_offroad:
            return -100.0  # Punição forte para o agente "sentir" o erro

        # 3. Recompensa de Estabilidade (Exponencial/Gaussiana)
        # Se estiver no centro (offset 0), ganha 1.0. 
        # Se estiver no limite (offset 2.0), ganha quase 0.
        reward = np.exp(-(lane_offset**2) / 0.5) 

        # 4. Bônus por Progresso (Já que a velocidade é constante)
        # Isso incentiva o agente a querer que o tempo passe enquanto ele está na pista
        reward += 0.5 

        # 5. Bônus de Sucesso (Se completar a distância configurada)
        if done and not is_offroad:
            reward += 50.0

        return float(reward)

    # def compute(self, observation: Dict, action: int, done: bool) -> float:
    #     reward = 0.0
    #     lane_offset = abs(observation.get("lane_offset", 0.0))
    #     heading_error = abs(observation.get("heading_error", 0.0))
    #     is_offroad = observation.get("offroad", False)

    #     reward -= lane_offset * self.lane_center_penalty
    #     reward -= heading_error * self.heading_penalty
    #     if is_offroad:
    #         reward -= self.offroad_penalty
    #     reward += self.speed_reward * observation.get("speed", 0.0)
    #     speed = observation.get("speed", 0.0)
    #     if speed < self.speed_penalty_threshold:
    #         reward -= (self.speed_penalty_threshold - speed) * self.speed_penalty_weight
    #     if done and not is_offroad:
    #         reward += 10.0
    #     return reward

