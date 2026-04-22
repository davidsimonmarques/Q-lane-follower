#!/usr/bin/env python3
"""
Script de benchmark para testar performance das otimizações.
Executa alguns episódios curtos e mede o tempo.
"""

import time
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import CONFIG
from environment.carla_env import CarlaEnvironment
from agent.q_learning_agent import QLearningAgent
from state.state_discretizer import StateDiscretizer


def benchmark_performance():
    """Executa benchmark de performance com episódios curtos."""
    print("🚀 Iniciando benchmark de performance...")
    print("=" * 50)

    # Configurações de benchmark (episódios mais curtos)
    benchmark_config = CONFIG.copy()
    benchmark_config["episodes"] = 5  # Apenas 5 episódios para teste
    benchmark_config["success_distance"] = 500  # Distância menor para episódios mais rápidos

    # Inicializar componentes
    env = CarlaEnvironment(benchmark_config)
    discretizer = StateDiscretizer(benchmark_config["discretizer_bins"])
    state_size = discretizer.state_space_size()
    agent = QLearningAgent(state_size, benchmark_config["action_size"], benchmark_config)

    total_time = 0
    total_steps = 0
    episode_times = []

    try:
        for episode in range(benchmark_config["episodes"]):
            start_time = time.time()

            # Reset do episódio
            observation = env.reset()
            state = discretizer.discretize(observation)
            done = False
            episode_steps = 0

            while not done and episode_steps < 1000:  # Limitar steps por episódio
                action = agent.choose_action(state)
                observation, _, done, _ = env.step(action)
                next_state = discretizer.discretize(observation)

                # Recompensa simples para benchmark
                reward = 1.0 if not done else 10.0
                agent.update(state, action, reward, next_state, done)

                state = next_state
                episode_steps += 1

            episode_time = time.time() - start_time
            episode_times.append(episode_time)
            total_time += episode_time
            total_steps += episode_steps

            print(f"Episode {episode+1}: {episode_steps} steps in {episode_time:.2f}s ({episode_steps/episode_time:.1f} steps/s)")
        # Estatísticas finais
        avg_episode_time = total_time / len(episode_times)
        avg_steps_per_second = total_steps / total_time

        print("\n" + "=" * 50)
        print("📊 RESULTADOS DO BENCHMARK:")
        print(f"Tempo total: {total_time:.2f}s")
        print(f"Total de steps: {total_steps}")
        print(f"Steps/segundo médio: {avg_steps_per_second:.1f}")
        print(f"Tempo médio por episódio: {avg_episode_time:.2f}s")
        print(f"Steps/episódio médio: {total_steps/len(episode_times):.0f}")

        # Avaliação de performance
        if avg_steps_per_second > 50:
            print("✅ Performance EXCELENTE! (>50 steps/seg)")
        elif avg_steps_per_second > 30:
            print("👍 Performance BOA! (30-50 steps/seg)")
        elif avg_steps_per_second > 15:
            print("⚠️  Performance RAZOÁVEL (15-30 steps/seg)")
        else:
            print("❌ Performance BAIXA (<15 steps/seg) - pode melhorar")

    except Exception as e:
        print(f"❌ Erro durante benchmark: {e}")
    finally:
        env.shutdown()

    return avg_steps_per_second


if __name__ == "__main__":
    benchmark_performance()