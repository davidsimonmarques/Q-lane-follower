# Data Logger Module

Este módulo fornece logging detalhado dos dados de treinamento em formato CSV.

## Funcionalidades

- **Logging por Episódio**: Salva dados detalhados de cada episódio
- **Médias Automáticas**: Calcula médias de lane offset, heading error e velocidade
- **Estatísticas**: Registra velocidade máxima/mínima por episódio
- **Resumo Final**: Gera estatísticas agregadas do treinamento

## Dados Registrados

| Coluna | Descrição |
|--------|-----------|
| episode | Número do episódio |
| total_reward | Recompensa total acumulada |
| avg_lane_offset | Média do offset da pista |
| avg_heading_error | Média do erro de direção |
| avg_speed | Velocidade média (m/s) |
| max_speed | Velocidade máxima |
| min_speed | Velocidade mínima |
| distance_traveled | Distância total percorrida |
| success | Se completou com sucesso |

## Uso

```python
from utils.data_logger import DataLogger

# Inicializar logger
logger = DataLogger()

# Log de episódio
logger.log_episode(
    episode=1,
    total_reward=15.5,
    lane_offsets=[0.1, 0.2, 0.1],
    heading_errors=[0.05, 0.1, 0.08],
    speeds=[5.0, 6.0, 5.5],
    distance_traveled=150.0,
    success=True
)

# Obter resumo
summary = logger.get_summary()
print(f"Taxa de sucesso: {summary['success_rate']:.1%}")
```

## Arquivos Gerados

- `logs/training_data.csv`: Dados detalhados por episódio
- Diretório `logs/` criado automaticamente