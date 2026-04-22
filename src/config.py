"""Configurações e hiperparâmetros do experimento."""

CONFIG = {
    "host": "127.0.0.1",
    "port": 2000,
    "episodes": 1000,
    "action_size": 7,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.3,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
    "log_level": "INFO",
    "q_table_path": "assets/q_table.npy",
    
    "load_pretrained": False,  # Se True: carrega Q-table existente. Se False: começa do zero

    # CONFIGURAÇÕES DE CAMERA E RENDER
    "render": False, #render do cv2 - camera frontal
    "no_rendering": True,  # Desabilitar renderização CARLA para treinamento mais rápido
    "success_distance": 2000,  # Distância em metros para sucesso (2km)
    "speed_penalty_threshold": 6,  # Velocidade mínima em m/s para evitar penalidade
    "speed_penalty_weight": 0.0,  # Peso da penalidade por velocidade baixa
    "camera_width": 800,
    "camera_height": 400,
    "camera_fov": 90,
    "camera_x": 1.5,
    "camera_z": 1.4,
    "top_down_height": 20.0,
    "top_down_pitch": -90.0,

    # OTIMIZAÇÕES DE PERFORMANCE
    "synchronous": False,  # Modo assíncrono para treinamento mais rápido
    "fixed_delta_seconds": 0.1,  # Timestep maior para menos cálculos de física
    "disable_camera": True,  # Desabilitar câmera completamente durante treinamento
    "disable_collision_sensor": False,  # Manter sensor de colisão (importante)
    "max_fps": 30,  # Limitar FPS para reduzir carga de CPU

    "action_map": {
        0: {"throttle": 0.1, "steer": -1.0, "brake": 0.0}, # Esquerda Máxima
        1: {"throttle": 0.2, "steer": -0.5, "brake": 0.0}, # Esquerda Forte
        2: {"throttle": 0.2, "steer": -0.1, "brake": 0.0}, # Esquerda Suave
        3: {"throttle": 0.3, "steer":  0.0, "brake": 0.0}, # Centro (Reto)
        4: {"throttle": 0.2, "steer":  0.1, "brake": 0.0}, # Direita Suave
        5: {"throttle": 0.2, "steer":  0.5, "brake": 0.0}, # Direita Forte
        6: {"throttle": 0.1, "steer":  1.0, "brake": 0.0}, # Direita Máxima
    },

    "discretizer_bins": {
        "lane_offset": [-0.5, -0.2, 0.0, 0.2, 0.5],
        "heading_error": [-0.4, -0.2, 0.0, 0.2, 0.4],
        "speed": [2.0, 4.0, 6.0, 8.0, 10.0]
    },
    # "map_name": "Town06"
}
