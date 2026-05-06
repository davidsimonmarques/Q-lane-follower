"""Cálculo de recompensa específico para lane following."""

from typing import Dict
import numpy as np

class RewardFunction:
    def __init__(self, config: Dict):
        self.lane_center_penalty = config.get("lane_center_penalty", 1.0)
        self.heading_penalty = config.get("heading_penalty", 0.5)
        self.offroad_penalty = config.get("offroad_penalty", 5.0)
        self.speed_reward = config.get("speed_reward", 0.15)

    def compute(self, observation: Dict, action: int, done: bool) -> float:
        lane_offset = abs(observation.get("lane_offset", 0.0))
        is_offroad = observation.get("offroad", False) or (lane_offset > 2.0)
        
        if is_offroad:
            # Penalidade única por sair da pista (não acumula por steps)
            return -10.0
        
        # Recompensa por manter-se na pista (maior no centro)
        reward = 1.0 - (lane_offset / 2.0)  # 1.0 no centro, 0 nas bordas
        # reward -= self.heading_penalty * abs(observation.get("heading_error", 0.0))  # Penalidade por erro de direção
        # Bônus de velocidade (substituindo o bônus de progresso)
        # Normalizado entre 0 e 0.15. Consideramos 10 m/s (~36 km/h) como referência.
        speed = observation.get("speed", 0.0)
        max_speed_ref = 10.0
        speed_bonus = min(self.speed_reward , (speed / max_speed_ref) * self.speed_reward)
        reward += speed_bonus
        
        # Bônus de sucesso (apenas quando completa a distância)
        if done and not is_offroad:
            reward += 5.0
        
        return float(reward)


    # def compute(self, observation: Dict, action: int, done: bool) -> float:
    #     # 1. Configurações de limites (pode colocar no config)
    #     MAX_OFFSET = 2.0  # Distância máxima da linha antes de ser considerado fora
        
    #     lane_offset = abs(observation.get("lane_offset", 0.0))
    #     is_offroad = observation.get("offroad", False) or (lane_offset > MAX_OFFSET)

    #     # 2. Se bateu ou saiu da pista: Punição Crítica e Fim
    #     if is_offroad:
    #         return -100.0  # Punição forte para o agente "sentir" o erro

    #     # 3. Recompensa de Estabilidade (Exponencial/Gaussiana)
    #     # Se estiver no centro (offset 0), ganha 1.0. 
    #     # Se estiver no limite (offset 2.0), ganha quase 0.
    #     reward = np.exp(-(lane_offset**2) / 0.5) 

    #     # 4. Bônus por Progresso (Já que a velocidade é constante)
    #     # Isso incentiva o agente a querer que o tempo passe enquanto ele está na pista
    #     reward += 0.5 

    #     # 5. Bônus de Sucesso (Se completar a distância configurada)
    #     if done and not is_offroad:
    #         reward += 50.0

    #     return float(reward)


