"""Avaliação do modelo treinado em lane following."""

import random

import carla
import pygame
import numpy as np
import math
import os
import sys
import logging
from datetime import datetime

# Suprimir logs verbosos do CARLA
logging.getLogger('carla').setLevel(logging.CRITICAL)
logging.getLogger('pygame').setLevel(logging.CRITICAL)

# Importar módulos do projeto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from agent.q_learning_agent import QLearningAgent
from state.state_discretizer import StateDiscretizer
from reward.reward_function import RewardFunction


class EvaluationConfig:
    """Configurações para avaliação do modelo."""
    
    def __init__(self):
        # CARLA
        self.host = "127.0.0.1"
        self.port = 2000
        
        # Otimização de Performance
        self.synchronous = True
        self.fixed_delta_seconds = 0.1
        self.disable_camera = False
        self.max_fps = 60
        
        # Visualização
        self.display_width = 1280
        self.display_height = 720
        
        # Modelo
        self.q_table_path = "assets/q_table.npy"
        self.load_pretrained = True
        
        # Estado discretizado
        self.discretizer_bins = CONFIG["discretizer_bins"]
        
        # Action map
        self.action_map = CONFIG["action_map"]
        # Duração
        self.max_steps = 2000
        self.success_distance = 2000


class CameraManager:
    """Gerencia câmera traseira."""
    
    def __init__(self, vehicle, world, width=1280, height=720):
        self.vehicle = vehicle
        self.world = world
        self.width = width
        self.height = height
        self.surface = None
        self.sensor = None
        self._setup_camera()
    
    def _setup_camera(self):
        """Configura câmera RGB traseira."""
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.width))
        blueprint.set_attribute('image_size_y', str(self.height))
        
        # Câmera traseira (vista por trás do carro)
        cam_transform = carla.Transform(
            carla.Location(x=-4, z=2.5),
            carla.Rotation(pitch=0)
        )
        self.sensor = self.world.spawn_actor(
            blueprint,
            cam_transform,
            attach_to=self.vehicle
        )
        self.sensor.listen(self._process_image)
    
    def _process_image(self, image):
        """Converte imagem CARLA para pygame."""
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def render(self, display):
        """Renderiza câmera."""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    def destroy(self):
        """Destrói câmera."""
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


class HUD:
    """HUD similar ao manual_control.py."""
    
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self._font_mono = pygame.font.Font(None, 24)
        self._info_text = []
    
    def tick(self, vehicle, distance, steps, action, observation, speed_kmh):
        """Atualiza informações do HUD."""
        loc = vehicle.get_location()
        rot = vehicle.get_transform().rotation
        
        action_names = ["L-Max", "L-Hard", "L-Soft", "Straight", "R-Soft", "R-Hard", "R-Max"]
        
        self._info_text = [
            f'Step:     {steps:6d}',
            f'Distance: {distance:8.1f} m',
            '',
            f'Speed:    {speed_kmh:6.1f} km/h',
            f'Location: ({loc.x:7.1f}, {loc.y:7.1f})',
            f'Height:   {loc.z:7.1f} m',
            f'Yaw:      {rot.yaw:7.1f}°',
            '',
            f'Lane Off: {observation.get("lane_offset", 0.0):7.3f} m',
            f'Head Err: {observation.get("heading_error", 0.0):7.3f} rad',
            f'Offroad:  {"YES" if observation.get("offroad", False) else "NO"}',
            f'Action:   {action_names[action]}',
        ]
    
    def render(self, display):
        """Renderiza painel HUD."""
        # Painel semi-transparente
        panel_width = 250
        panel = pygame.Surface((panel_width, self.dim[1]))
        panel.set_alpha(180)
        panel.fill((0, 0, 0))
        display.blit(panel, (0, 0))
        
        # Renderizar texto
        v_offset = 10
        for line in self._info_text:
            if line:
                text_surface = self._font_mono.render(line, True, (255, 255, 255))
                display.blit(text_surface, (10, v_offset))
            v_offset += 28


def get_observation(vehicle, world):
    """Cria observação do estado atual compatível com o treinamento."""
    transform = vehicle.get_transform()
    location = transform.location
    velocity = vehicle.get_velocity()
    
    # 1. Velocidade escalar
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # 2. Obter o waypoint mais próximo na pista
    waypoint = world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    lane_offset = 0.0
    heading_error = 0.0
    offroad = True

    if waypoint:
        # 3. Cálculo do Lane Offset (Distância perpendicular real)
        # Usamos o vetor de direção do waypoint para projetar o desvio lateral
        w_trans = waypoint.transform
        dx = location.x - w_trans.location.x
        dy = location.y - w_trans.location.y
        
        # Converte o ângulo do waypoint para radianos
        yaw_rad = math.radians(w_trans.rotation.yaw)
        
        # Projeção lateral (Fórmula que você usa no wrapper)
        side_offset = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        lane_offset = float(side_offset)
        
        # 4. Cálculo do Heading Error (Normalizado)
        # Diferença angular entre o carro e a estrada entre -180 e 180
        error = (w_trans.rotation.yaw - transform.rotation.yaw + 180.0) % 360.0 - 180.0
        heading_error = float(error)
        
        offroad = False

    return {
        "location": location,
        "speed": speed,
        "lane_offset": lane_offset,
        "heading_error": heading_error,
        "offroad": offroad,
    }



def main():
    """Loop principal de avaliação."""
    config = EvaluationConfig()
    
    pygame.init()
    pygame.font.init()
    
    display = pygame.display.set_mode(
        (config.display_width, config.display_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    display.fill((0, 0, 0))
    pygame.display.set_caption("Q-Lane Follower Evaluation")
    pygame.display.flip()
    
    clock = pygame.time.Clock()
    
    # Conectar ao CARLA
    print("Conectando ao CARLA...")
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Configurar modo síncrono/assíncrono
    settings = world.get_settings()
    settings.synchronous_mode = config.synchronous
    settings.fixed_delta_seconds = config.fixed_delta_seconds
    if config.max_fps:
        settings.max_substep_delta_time = 1.0 / config.max_fps
        settings.max_substeps = 1
    world.apply_settings(settings)
    
    vehicle = None
    camera_manager = None
    
    try:
        # Inicializar ambiente CARLA
        print("Inicializando ambiente...")
        blueprint_lib = world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Nenhum spawn point disponível")
        
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)
        
        # Inicializar discretizador
        discretizer = StateDiscretizer(config.discretizer_bins)
        state_size = discretizer.state_space_size()
        
        # Carregar modelo treinado
        print(f"Carregando modelo de {config.q_table_path}...")
        if not os.path.exists(config.q_table_path):
            raise FileNotFoundError(f"Modelo não encontrado em {config.q_table_path}")
        
        agent = QLearningAgent(state_size, len(config.action_map), {
            "q_table_path": config.q_table_path,
            "load_pretrained": config.load_pretrained,
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon": 0.0  # Sem exploração, apenas explotação
        })
        
        # Criar câmera
        print("Configurando câmera...")
        camera_manager = CameraManager(vehicle, world, config.display_width, config.display_height)
        
        # Criar HUD
        hud = HUD(config.display_width, config.display_height)
        
        # Referência para o Spectator (Câmera do carla.exe)
        spectator = world.get_spectator()
        
        # Observação inicial
        print("Iniciando avaliação...")
        
        distance_traveled = 0.0
        previous_location = vehicle.get_location()
        steps = 0
        done = False
        
        start_time = datetime.now()
        
        while not done:# and steps < config.max_steps:
            # Input de controle (ESC para sair)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            
            # Obter observação
            observation = get_observation(vehicle, world)
            
            # Agente escolhe ação (explotação pura)
            state = discretizer.discretize(observation)
            action = agent.choose_action(state)
            
            # Aplicar ação
            control = carla.VehicleControl()
            action_data = config.action_map[action]
            control.throttle = action_data["throttle"]
            control.steer = action_data["steer"]
            control.brake = action_data["brake"]
            vehicle.apply_control(control)
            
            # Tick do mundo
            if config.synchronous:
                world.tick()
            else:
                world.wait_for_tick()
            
            # Atualizar a visão do spectator no carla.exe (visão superior)
            transform = vehicle.get_transform()
            position = transform.location + carla.Location(z=40.0)
            rotation = carla.Rotation(pitch=-90.0, yaw=0.0)
            spectator.set_transform(carla.Transform(position, rotation))
            
            # Atualizar observação
            observation = get_observation(vehicle, world)
            current_location = vehicle.get_location()
            
            # Calcular distância percorrida
            dx = current_location.x - previous_location.x
            dy = current_location.y - previous_location.y
            distance = math.sqrt(dx*dx + dy*dy)
            distance_traveled += distance
            previous_location = current_location
            
            # Verificar se chegou ao fim
            done = observation.get("offroad", False) or distance_traveled >= config.success_distance
            
            # Calcular speed em km/h
            speed = observation["speed"]
            speed_kmh = speed * 3.6
            
            steps += 1
            
            # Renderizar
            display.fill((0, 0, 0))
            camera_manager.render(display)
            hud.tick(vehicle, distance_traveled, steps, action, observation, speed_kmh)
            hud.render(display)
            
            # FPS
            fps = clock.get_fps()
            fps_text = pygame.font.Font(None, 16).render(f"FPS: {fps:.1f}", True, (0, 255, 0))
            display.blit(fps_text, (10, config.display_height - 20))
            
            pygame.display.flip()
            clock.tick(60)
        
        # Resultado final
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n=== AVALIAÇÃO CONCLUÍDA ===")
        print(f"Tempo: {elapsed:.2f}s")
        print(f"Steps: {steps}")
        print(f"Distância: {distance_traveled:.2f}m")
        print(f"Sucesso: {'SIM' if distance_traveled >= config.success_distance else 'NÃO'}")
        
        # Manter tela visível por 3 segundos
        for _ in range(180):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
            pygame.display.flip()
            clock.tick(60)
    
    except Exception as e:
        print(f"Erro durante avaliação: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpeza
        try:
            if camera_manager:
                camera_manager.destroy()
            if vehicle:
                vehicle.destroy()
        except Exception as e:
            print(f"Erro durante limpeza: {e}")
        
        pygame.quit()


if __name__ == "__main__":
    main()
