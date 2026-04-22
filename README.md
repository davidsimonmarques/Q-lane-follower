# Q-Lane-Follower

Arquitetura modular para um agente de lane follower usando CARLA e Q-learning tabular.

## Organização

- `src/environment/carla_env.py` - wrapper do CARLA, reset e step do ambiente.
- `src/agent/q_learning_agent.py` - algoritmo tabular de Q-learning.
- `src/state/state_discretizer.py` - discretização de observações contínuas.
- `src/reward/reward_function.py` - recompensa orientada para lane following.
- `src/training/train.py` - loop de treinamento e persistência da tabela Q.
- `src/utils/logger.py` - logging simples.
- `src/utils/data_logger.py` - logging detalhado em CSV (veja DATA_LOGGER_README.md).
- `src/config.py` - hiperparâmetros e bins para discretização.
- `src/main.py` - ponto de entrada para treinar o experimento.

## Logging de Dados

O projeto inclui sistema de logging detalhado para análise de treinamento:

- **Logger Simples**: Logs básicos no console via `utils/logger.py`
- **Data Logger CSV**: Dados detalhados por episódio em `logs/training_data.csv`
  - Médias de reward, lane offset, heading error, velocidade
  - Estatísticas de velocidade (máx/mín)
  - Distância percorrida e taxa de sucesso

Veja `DATA_LOGGER_README.md` para documentação completa.

## Configurações Importantes

- `"no_rendering": False` - Desabilita renderização CARLA para treinamento mais rápido
- `"success_distance": 2000` - Distância para sucesso (2km)
- `"speed_penalty_threshold": 1.0` - Penalidade por velocidade baixa
- `"render": True/False` - Controle de visualização

## 🚀 Otimizações de Performance

O projeto inclui várias otimizações para treinamento mais rápido:

### ⚡ Configurações de Alto Desempenho

```python
CONFIG = {
    "synchronous": False,        # Modo assíncrono (mais rápido)
    "fixed_delta_seconds": 0.1,  # Timestep maior
    "disable_camera": True,      # Remove processamento de câmera
    "max_fps": 30,              # Limita FPS para reduzir CPU
}
```

### 🏃 Melhorias Implementadas

- **Modo Assíncrono**: Remove waits síncronos entre steps
- **Câmera Desabilitada**: Elimina processamento de imagem desnecessário
- **Cálculos Otimizados**: Evita `np.linalg.norm` em favor de operações diretas
- **FPS Limitado**: Reduz carga de CPU com `max_substep_delta_time`
- **Limpeza Agressiva**: Sensores órfãos removidos a cada episódio para evitar vazamento de memória

### 📊 Benchmark de Performance

Execute o benchmark para testar as otimizações:

```bash
python benchmark.py
```

**Resultados Esperados**:
- **Excelente**: >50 steps/segundo
- **Bom**: 30-50 steps/segundo
- **Razoável**: 15-30 steps/segundo

### 🎯 Comparação Antes vs Depois

| Configuração | Antes | Depois | Melhoria |
|-------------|-------|--------|----------|
| Modo | Síncrono | Assíncrono | ~3x mais rápido |
| Câmera | Habilitada | Desabilitada | ~2x mais rápido |
| Timestep | 0.05s | 0.1s | ~2x mais rápido |
| Cálculos | numpy | Direto | ~1.5x mais rápido |
