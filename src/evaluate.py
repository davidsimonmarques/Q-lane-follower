"""Avaliação do modelo treinado em lane following."""

import random

import carla
import pygame
import numpy as np
import math
import os
import cv2
import sys
import logging
from datetime import datetime
import csv
from typing import Optional

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
        self.fixed_delta_seconds = 0.01
        self.disable_camera = False
        self.max_fps = 60
        self.no_rendering = True
        
        # Visualização
        self.display_width = 1280
        self.display_height = 720
        
        # Modelo
        self.q_table_path = "assets/q_table.npy" #"assets/q_table.npy"
        self.load_pretrained = True
        
        # Estado discretizado
        self.discretizer_bins = CONFIG["discretizer_bins"]
        
        # Action map
        self.action_map = CONFIG["action_map"]
        # Duração
        self.max_steps = 2000
        self.success_distance = 2000
        
        # Gravação
        self.record_pygame = True
        self.record_carla = True
        self.pygame_video_path = "evaluation_pygame.mp4"
        self.carla_rec_path = os.path.abspath("evaluation_carla.log").replace('\\', '/')

class EvaluationDataLogger:
    """Logger para salvar dados de avaliação passo a passo em CSV."""

    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        self.log_dir = log_dir
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_data_{timestamp}.csv"
        
        self.filepath = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)

        self.csv_file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        
        # Escrever cabeçalho
        self.writer.writerow([
            'timestamp_ms',
            'step',
            'speed_ms',
            'speed_kmh',
            'lane_offset',
            'heading_error',
            'action',
            'throttle',
            'steer',
            'brake',
            'distance_traveled',
            'offroad'
        ])

    def log_step(self, timestamp_ms: float, step: int, observation: dict, action: int, control: carla.VehicleControl, distance_traveled: float):
        """Registra os dados de um único passo de simulação."""
        self.writer.writerow([
            int(timestamp_ms),
            step,
            f"{observation.get('speed', 0.0):.4f}",
            f"{observation.get('speed', 0.0) * 3.6:.4f}",
            f"{observation.get('lane_offset', 0.0):.4f}",
            f"{observation.get('heading_error', 0.0):.4f}",
            action,
            f"{control.throttle:.4f}",
            f"{control.steer:.4f}",
            f"{control.brake:.4f}",
            f"{distance_traveled:.4f}",
            observation.get('offroad', False)
        ])

    def close(self):
        """Fecha o arquivo CSV."""
        if self.csv_file:
            self.csv_file.close()
            print(f"\nLog de avaliação salvo em: {self.filepath}")

class CameraManager:
    """Gerencia a câmera de visualização."""
    
    def __init__(self, vehicle, world, width=1280, height=720, transform=None, attachment_type=carla.AttachmentType.SpringArmGhost):
        self.vehicle = vehicle
        self.world = world
        self.width = width
        self.height = height
        self.transform = transform
        self.attachment_type = attachment_type
        self.surface = None
        self.sensor = None
        self._setup_camera()
    
    def _setup_camera(self):
        """Configura a câmera RGB."""
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.width))
        blueprint.set_attribute('image_size_y', str(self.height))
        
        if self.transform is None:
            bound_x = 0.5 + self.vehicle.bounding_box.extent.x
            bound_y = 0.5 + self.vehicle.bounding_box.extent.y
            bound_z = 0.5 + self.vehicle.bounding_box.extent.z
            
            # Usando a câmera traseira
            self.transform = carla.Transform(
                carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z),
                carla.Rotation(pitch=8.0)
            )
            
        self.sensor = self.world.spawn_actor(
            blueprint,
            self.transform,
            attach_to=self.vehicle,
            attachment_type=self.attachment_type
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
    
    def render(self, display, pos=(0, 0)):
        """Renderiza câmera."""
        if self.surface is not None:
            display.blit(self.surface, pos)
    
    def destroy(self):
        """Destrói câmera."""
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


class HUD:
    """HUD similar ao manual_control.py."""
    
    def __init__(self, width, height):
        self.dim = (width, height)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._info_text = []
    
    def tick(self, vehicle, distance, steps, action, observation, speed_kmh):
        """Atualiza informações do HUD."""
        loc = vehicle.get_location()
        rot = vehicle.get_transform().rotation
        control = vehicle.get_control()
        
        action_names = ["L-Max", "L-Hard", "L-Soft", "Straight", "R-Soft", "R-Hard", "R-Max"]
        
        self._info_text = [
            'Step:    % 7d' % steps,
            'Distance:% 8.1f m' % distance,
            '',
            'Speed:   % 7.1f km/h' % speed_kmh,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (loc.x, loc.y)),
            'Height:  % 7.1f m' % loc.z,
            'Yaw:     % 7.1f deg' % rot.yaw,
            '',
            'Lane Off:% 7.3f m' % observation.get("lane_offset", 0.0),
            'Head Err:% 7.3f rad' % observation.get("heading_error", 0.0),
            'Offroad: % 10s' % ("YES" if observation.get("offroad", False) else "NO"),
            'Action:  % 10s' % action_names[action],
            '',
            ('Throttle:', control.throttle, 0.0, 1.0),
            ('Steer:', control.steer, -1.0, 1.0)
        ]
    
    def render(self, display):
        """Renderiza painel HUD."""
        info_surface = pygame.Surface((220, self.dim[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset = 4
        bar_h_offset = 100
        bar_width = 106
        for item in self._info_text:
            if v_offset + 18 > self.dim[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    fig = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect(
                            (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18


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


def generate_map_images(world, trajectory, base_name="evaluation_result"):
    """Gera e salva a visão superior do mapa e o trajeto percorrido."""
    try:
        print("\nGerando imagens do mapa. Aguarde...")
        
        # Pular os primeiros 10 frames da trajetória (ignora solavancos iniciais do spawn)
        if trajectory and len(trajectory) > 10:
            trajectory = trajectory[10:]
            
        # Pegar waypoints para desenhar o mapa (espaçamento de 2.0 metros)
        waypoints = world.get_map().generate_waypoints(2.0)
        
        # Encontrar limites do mapa
        min_x = min(wp.transform.location.x for wp in waypoints)
        max_x = max(wp.transform.location.x for wp in waypoints)
        min_y = min(wp.transform.location.y for wp in waypoints)
        max_y = max(wp.transform.location.y for wp in waypoints)
        
        # Considerar também os limites da trajetória
        if trajectory:
            min_x = min(min_x, min(p[0] for p in trajectory))
            max_x = max(max_x, max(p[0] for p in trajectory))
            min_y = min(min_y, min(p[1] for p in trajectory))
            max_y = max(max_y, max(p[1] for p in trajectory))
            
        # Adicionar margem de respiro
        padding = 50.0
        min_x -= padding; max_x += padding
        min_y -= padding; max_y += padding
        
        # Calcular dimensões da imagem (max 1500 pixels de largura/altura)
        width_m = max_x - min_x
        height_m = max_y - min_y
        scale = 1500.0 / max(width_m, height_m, 1.0)
        
        img_w = int(width_m * scale)
        img_h = int(height_m * scale)
        
        # Criar imagem base (fundo branco)
        map_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        def to_pixel(x, y):
            px = int((x - min_x) * scale)
            py = int((y - min_y) * scale)
            return (px, py)
            
        # Desenhar eixos / grid de referência (a cada 100 metros)
        grid_step = 100
        
        # Linhas Verticais (Eixo X)
        start_x = int(math.ceil(min_x / grid_step)) * grid_step
        for x_m in range(start_x, int(max_x), grid_step):
            px, _ = to_pixel(x_m, min_y)
            cv2.line(map_img, (px, 0), (px, img_h), (0, 0, 0), 1)
            cv2.putText(map_img, f"{x_m}m", (px + 5, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        # Linhas Horizontais (Eixo Y)
        start_y = int(math.ceil(min_y / grid_step)) * grid_step
        for y_m in range(start_y, int(max_y), grid_step):
            _, py = to_pixel(min_x, y_m)
            cv2.line(map_img, (0, py), (img_w, py), (0, 0, 0), 1)
            cv2.putText(map_img, f"{y_m}m", (15, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        # Desenhar ruas
        for wp in waypoints:
            px, py = to_pixel(wp.transform.location.x, wp.transform.location.y)
            cv2.circle(map_img, (px, py), max(int(2 * scale), 1), (60, 60, 60), -1)
            
        # Salvar mapa base
        cv2.imwrite(f"{base_name}_map.png", map_img)
        
        # Desenhar trajetória
        traj_img = map_img.copy()
        if len(trajectory) > 1:
            # Desenhar linha do trajeto
            for i in range(len(trajectory) - 1):
                p1 = to_pixel(trajectory[i][0], trajectory[i][1])
                p2 = to_pixel(trajectory[i+1][0], trajectory[i+1][1])
                # OpenCV usa BGR. Laranja = (0, 150, 255)
                cv2.line(traj_img, p1, p2, (0, 150, 255), max(int(1 * scale), 2))
            
            # Desenhar setas indicativas de direção ao longo do trajeto (a cada 10 metros)
            dist_accum = 0.0
            for i in range(1, len(trajectory) - 1):
                dx = trajectory[i][0] - trajectory[i-1][0]
                dy = trajectory[i][1] - trajectory[i-1][1]
                dist_accum += math.hypot(dx, dy)
                
                if dist_accum >= 10.0:
                    p1 = to_pixel(trajectory[i][0], trajectory[i][1])
                    # Buscar um ponto mais à frente para garantir um tamanho legível da seta na imagem
                    for j in range(i + 1, len(trajectory)):
                        p2 = to_pixel(trajectory[j][0], trajectory[j][1])
                        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 15:  # Pelo menos 15 pixels de tamanho
                            cv2.arrowedLine(traj_img, p1, p2, (0, 150, 255), max(int(1.5 * scale), 2), tipLength=0.4)
                            break
                    dist_accum = 0.0
            
            # Ponto Inicial (Verde)
            p_start = to_pixel(trajectory[0][0], trajectory[0][1])
            cv2.circle(traj_img, p_start, max(int(2 * scale), 4), (0, 255, 0), -1)
            # cv2.putText(traj_img, "INICIO", (p_start[0] + 10, p_start[1] - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Ponto Final (Vermelho)
            p_end = to_pixel(trajectory[-1][0], trajectory[-1][1])
            cv2.circle(traj_img, p_end, max(int(2 * scale), 4), (0, 0, 255), -1)
            # cv2.putText(traj_img, "FIM", (p_end[0] + 10, p_end[1] - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        cv2.imwrite(f"{base_name}_trajectory.png", traj_img)
        print(f"-> Imagens salvas: {base_name}_map.png e {base_name}_trajectory.png")
        
    except Exception as e:
        print(f"Erro ao gerar imagens do mapa: {e}")


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
    
    # Iniciar gravador CARLA (salva um .log para replay)
    carla_recorder_started = False
    if config.record_carla:
        try:
            client.start_recorder(config.carla_rec_path)
            carla_recorder_started = True
            print(f"Gravador CARLA iniciado. Salvando em: {config.carla_rec_path}")
        except Exception as e:
            print(f"Não foi possível iniciar o gravador CARLA: {e}")
            
    world = client.get_world()
    
    # Configurar modo síncrono/assíncrono
    settings = world.get_settings()
    settings.synchronous_mode = config.synchronous
    settings.fixed_delta_seconds = config.fixed_delta_seconds
    settings.no_rendering_mode = config.no_rendering
    if config.max_fps:
        settings.max_substep_delta_time = 1.0 / config.max_fps
        settings.max_substeps = 1
    world.apply_settings(settings)
    
    vehicle = None
    camera_manager = None
    bird_camera_manager = None
    pygame_video_writer = None
    evaluation_logger = None
    
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
        
        # Simular alguns frames silenciosamente para o carro assentar no chão
        for _ in range(20):
            if config.synchronous:
                world.tick()
            else:
                world.wait_for_tick()
        
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
        print("Configurando câmeras...")
        camera_manager = CameraManager(vehicle, world, config.display_width, config.display_height)
        
        # Criar câmera Bird View
        bird_w, bird_h = 320, 240
        bird_transform = carla.Transform(carla.Location(z=20.0), carla.Rotation(pitch=-90.0))
        bird_camera_manager = CameraManager(
            vehicle, world, bird_w, bird_h,
            transform=bird_transform,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # Criar HUD
        hud = HUD(config.display_width, config.display_height)
        
        # Iniciar gravador de vídeo Pygame (salva um .mp4)
        if config.record_pygame:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Usar o FPS do clock para o vídeo, com um padrão de 60
            fps = 60.0
            pygame_video_writer = cv2.VideoWriter(
                config.pygame_video_path, 
                fourcc, 
                fps, 
                (config.display_width, config.display_height)
            )
            print(f"Gravador Pygame iniciado. Salvando em: {config.pygame_video_path}")
            
        # Inicializar logger de avaliação
        evaluation_logger = EvaluationDataLogger()

        # Observação inicial
        print("Iniciando avaliação...")
        
        distance_traveled = 0.0
        previous_location = vehicle.get_location()
        steps = 0
        done = False
        
        # Rastrear trajetória para o mapa no final
        trajectory = [(previous_location.x, previous_location.y)]
        
        
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
            
            # Atualizar observação
            observation = get_observation(vehicle, world)
            current_location = vehicle.get_location()
            
            # Calcular distância percorrida
            dx = current_location.x - previous_location.x
            dy = current_location.y - previous_location.y
            distance = math.sqrt(dx*dx + dy*dy)
            distance_traveled += distance
            previous_location = current_location
            
            # Salvar ponto da trajetória
            trajectory.append((current_location.x, current_location.y))
            
            # Verificar se chegou ao fim
            done = observation.get("offroad", False) or distance_traveled >= config.success_distance
            
            # Calcular speed em km/h
            speed = observation["speed"]
            speed_kmh = speed * 3.6
            
            # Logar dados do passo
            snapshot = world.get_snapshot()
            sim_time_ms = snapshot.timestamp.elapsed_seconds * 1000
            if evaluation_logger:
                evaluation_logger.log_step(
                    sim_time_ms,
                    steps,
                    observation,
                    action,
                    control,
                    distance_traveled
                )

            steps += 1
            
            # Renderizar
            display.fill((0, 0, 0))
            camera_manager.render(display, pos=(0, 0))
            
            # Renderizar Bird View (com borda para destacar do fundo)
            bird_pos = (config.display_width - bird_w - 20, 20)
            pygame.draw.rect(display, (200, 200, 200), (bird_pos[0]-2, bird_pos[1]-2, bird_w+4, bird_h+4), 2)
            bird_camera_manager.render(display, pos=bird_pos)
            
            hud.tick(vehicle, distance_traveled, steps, action, observation, speed_kmh)
            hud.render(display)
            
            # FPS
            fps = clock.get_fps()
            fps_text = pygame.font.Font(None, 16).render(f"FPS: {fps:.1f}", True, (0, 255, 0))
            display.blit(fps_text, (10, config.display_height - 20))
            
            # Gravar frame do Pygame
            if pygame_video_writer:
                # Captura o frame do display do pygame
                frame = pygame.surfarray.array3d(display)
                # Converte de (width, height, RGB) para (height, width, RGB) e depois para BGR
                frame = cv2.cvtColor(frame.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
                pygame_video_writer.write(frame)
                
            pygame.display.flip()
            clock.tick(60)
        
        # Resultado final
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n=== AVALIAÇÃO CONCLUÍDA ===")
        print(f"Tempo: {elapsed:.2f}s")
        print(f"Steps: {steps}")
        print(f"Distância: {distance_traveled:.2f}m")
        print(f"Sucesso: {'SIM' if distance_traveled >= config.success_distance else 'NÃO'}")
        
        # Gerar imagens de resumo final do trajeto
        generate_map_images(world, trajectory)
        
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
        # Limpeza e SALVAR GRAVAÇÕES
        try:
            if evaluation_logger:
                evaluation_logger.close()

            if pygame_video_writer:
                pygame_video_writer.release()
                print(f"Vídeo Pygame salvo em: {config.pygame_video_path}")
            if carla_recorder_started:
                client.stop_recorder()
                print(f"Gravação CARLA salva em: {config.carla_rec_path}")
                
            if camera_manager:
                camera_manager.destroy()
            if bird_camera_manager:
                bird_camera_manager.destroy()
            if vehicle:
                vehicle.destroy()
        except Exception as e:
            print(f"Erro durante limpeza: {e}")
        
        pygame.quit()


if __name__ == "__main__":
    main()
