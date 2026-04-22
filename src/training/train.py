"""Loop de treinamento de Q-learning."""

import time
from typing import Dict

from environment.carla_env import CarlaEnvironment
from agent.q_learning_agent import QLearningAgent
from state.state_discretizer import StateDiscretizer
from reward.reward_function import RewardFunction
from utils.logger import setup_logger
from utils.data_logger import DataLogger


def train(config: Dict) -> None:
    logger = setup_logger(config)
    data_logger = DataLogger()
    
    # Inicializamos o ambiente fora do try para garantir que o objeto exista
    # mas o gerenciamento de atores ocorrerá dentro do bloco de proteção.
    env = CarlaEnvironment(config)
    
    try:
        discretizer = StateDiscretizer(config["discretizer_bins"])
        reward_fn = RewardFunction(config)

        state_size = discretizer.state_space_size()
        agent = QLearningAgent(state_size, config["action_size"], config)

        for episode in range(config.get("episodes", 100)):
            # O reset agora lida com a limpeza de veículos antigos internamente
            observation = env.reset()
            state = discretizer.discretize(observation)
            done = False
            episode_reward = 0.0
            
            # Listas para coletar dados do episódio
            lane_offsets = []
            heading_errors = []
            speeds = []

            while not done:
                action = agent.choose_action(state)
                
                # O ambiente executa a ação e o mundo dá um "tick"
                observation, _, done, info = env.step(action)
                
                # Coletar dados para logging
                lane_offsets.append(abs(observation.get("lane_offset", 0.0)))
                heading_errors.append(abs(observation.get("heading_error", 0.0)))
                speeds.append(observation.get("speed", 0.0))
                
                # Cálculo da recompensa baseado na nova observação
                reward = reward_fn.compute(observation, action, done)
                
                next_state = discretizer.discretize(observation)
                
                # Atualização da Q-Table
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward

                if config.get("render", False):
                    env.render()

            # Log dos dados do episódio
            distance_traveled = info.get("distance_traveled", 0.0)
            success = info.get("success", False)
            data_logger.log_episode(
                episode + 1,
                episode_reward,
                lane_offsets,
                heading_errors,
                speeds,
                distance_traveled,
                success
            )

            # Logs de fim de episódio
            if info.get("collision"):
                logger.warning(f"Episode {episode+1} ended by collision.")
            elif observation.get("offroad"):
                logger.warning(f"Episode {episode+1} ended by going offroad.")
            elif success:
                logger.info(f"Episode {episode+1} completed successfully! Distance: {distance_traveled:.2f}m")

            # Evolução do agente (exploração vs explotação)
            agent.decay_epsilon(
                config.get("min_epsilon", 0.01), 
                config.get("epsilon_decay", 0.995)
            )
            
            logger.info(f"Episode {episode+1}/{config.get('episodes')} reward={episode_reward:.2f}")

        # Salva o progresso após o sucesso de todos os episódios
        agent.save(config.get("q_table_path", "assets/q_table.npy"))
        
        # Exibir resumo final
        summary = data_logger.get_summary()
        if summary:
            logger.info("=== TRAINING SUMMARY ===")
            logger.info(f"Total episodes: {summary['total_episodes']}")
            logger.info(f"Average reward: {summary['avg_reward']:.2f}")
            logger.info(f"Success rate: {summary['success_rate']:.2%}")
            logger.info(f"Best reward: {summary['best_reward']:.2f}")
            logger.info(f"Worst reward: {summary['worst_reward']:.2f}")

    except KeyboardInterrupt:
        logger.info("Treinamento interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        logger.error(f"Erro crítico durante o treinamento: {e}")
        raise e # Re-levanta o erro para debug após a limpeza
    finally:
        # CRITICAL: Isso garante que o veículo e sensores sejam destruídos
        # mesmo se o código falhar ou for interrompido.
        logger.info("Encerrando conexão com CARLA e limpando atores...")
        env.shutdown()