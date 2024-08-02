import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pygame
from numba import jit, prange

pygame.init()

# Dimensões da janela
WIDTH, HEIGHT = 1200, 800
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 120
BUTTON_PADDING = 10
PADDING = 20
SLIDER_WIDTH = 200
SLIDER_HEIGHT = 20
SLIDER_PADDING = 10
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulação de Partículas - Pedro Henrique")

# Configurações das partículas
FORCE_DIFF = 2
AMPLIFIER = 1
PARTICLE_RADIUS = 4
SAFETY_MARGIN = 0
DAMPING = 0.98
COLLISION_DAMPING = 0.5
INTERACTION_RADIUS = 50

NUM_PARTICLE_TYPES = 5
NUM_PARTICLES = 3000

CLOCK = 2000

def generate_particles_and_interactions(custom_interaction_table=None):
    particle_types = {}
    for i in range(NUM_PARTICLE_TYPES):
        particle_type = chr(ord('A') + i)
        color = (
            random.randint(20, 200),  # R
            random.randint(20, 200),  # G
            random.randint(20, 200)   # B
        )
        particle_types[particle_type] = {'color': color}

    if custom_interaction_table is None or custom_interaction_table.size == 0:
        interaction_table = np.zeros((NUM_PARTICLE_TYPES, NUM_PARTICLE_TYPES), dtype=np.float64)
        for i in range(NUM_PARTICLE_TYPES):
            for j in range(NUM_PARTICLE_TYPES):
                interaction_table[i, j] = random.uniform(FORCE_DIFF * (-1), FORCE_DIFF)
    else:
        interaction_table = custom_interaction_table

    particles = np.zeros((NUM_PARTICLES, 6), dtype=np.float64)
    for i in range(NUM_PARTICLES):
        ptype = random.choice(list(particle_types.keys()))
        ptype_index = ord(ptype) - ord('A')
        x = random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        y = random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS - BUTTON_HEIGHT - PADDING)
        color = particle_types[ptype]['color']
        particles[i] = [x, y, 0, 0, ptype_index, color[0] * 65536 + color[1] * 256 + color[2]]
        
    print("\nTabela de Interações:")
    for i in range(NUM_PARTICLE_TYPES):
        print(f"Tipo {chr(ord('A') + i)}: {interaction_table[i]}")

    return particles, interaction_table


@jit(nopython=True, parallel=True)
def calculate_forces(particles, interaction_table, amplifier, collision_damping):
    num_particles = particles.shape[0]
    for i in prange(num_particles):
        p1 = particles[i]
        x1, y1, vx1, vy1, type_index1, _ = p1
        fx, fy = 0.0, 0.0
        for j in prange(num_particles):
            if i == j:
                continue
            p2 = particles[j]
            x2, y2, vx2, vy2, type_index2, _ = p2

            dx = x2 - x1
            dy = y2 - y1

            # Considerar bordas conectadas
            dx -= WIDTH * np.round(dx / WIDTH)
            dy -= HEIGHT * np.round(dy / HEIGHT)

            distance = np.hypot(dx, dy)
            if distance > INTERACTION_RADIUS:
                continue

            if distance > 0:
                interaction = interaction_table[int(type_index1), int(type_index2)]
                force = (interaction / distance**2) * amplifier
                fx += force * dx / distance
                fy += force * dy / distance

            # Ajustar colisões para evitar saltos grandes
            if distance < 2 * (PARTICLE_RADIUS + SAFETY_MARGIN):
                overlap = 2 * (PARTICLE_RADIUS + SAFETY_MARGIN) - distance
                if distance > 0:
                    overlap_fraction = overlap / distance
                    fx -= collision_damping * overlap_fraction * (dx / distance)
                    fy -= collision_damping * overlap_fraction * (dy / distance)

        # Aplicar as forças calculadas às partículas
        particles[i, 2] += fx
        particles[i, 3] += fy

        # Limitar a velocidade das partículas para evitar movimentos extremos
        max_speed = 18  # ajuste conforme necessário
        speed = np.hypot(particles[i, 2], particles[i, 3])
        if speed > max_speed:
            particles[i, 2] = (particles[i, 2] / speed) * max_speed
            particles[i, 3] = (particles[i, 3] / speed) * max_speed

    # Aplicar posições atualizadas e considerar bordas
    for i in prange(num_particles):
        particles[i, 0] = (particles[i, 0] + particles[i, 2]) % WIDTH
        particles[i, 1] = (particles[i, 1] + particles[i, 3]) % HEIGHT


@jit(nopython=True)
def update_particles(particles):
    particles[:, 2] *= DAMPING
    particles[:, 3] *= DAMPING
    particles[:, 0] += particles[:, 2]
    particles[:, 1] += particles[:, 3]

    particles[:, 0] = particles[:, 0] % WIDTH
    particles[:, 1] = particles[:, 1] % HEIGHT


def draw_particles(particles, win):
    for p in particles:
        x, y, _, _, _, color = p
        color = int(color)
        r = (color >> 16) & 255
        g = (color >> 8) & 255
        b = color & 255
        pygame.draw.circle(win, (r, g, b), (int(x), int(y)), PARTICLE_RADIUS)


def draw_button(win):
    button_color = (0, 128, 255)
    button_rect = pygame.Rect(WIDTH - BUTTON_WIDTH - BUTTON_PADDING,
                              HEIGHT - BUTTON_HEIGHT - BUTTON_PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(win, button_color, button_rect)

    font = pygame.font.Font(None, 36)
    text_surf = font.render('Reset', True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=button_rect.center)
    win.blit(text_surf, text_rect)

    return button_rect


def draw_slider(win, x, y, width, height, value, min_value, max_value, label):
    pygame.draw.rect(win, (100, 100, 100), (x, y, width, height))
    handle_x = int((value - min_value) / (max_value - min_value) * width) + x
    pygame.draw.rect(win, (200, 200, 200), (handle_x - 10, y - 10, 20, height + 10))
    
    font = pygame.font.Font(None, 24)
    text_surf = font.render(f'{label}: {value:.2f}', True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(x + width / 2, y - 15))
    win.blit(text_surf, text_rect)

def main():
    global FORCE_DIFF, AMPLIFIER, COLLISION_DAMPING

    # CELULAS EXPLOSIVAS
    custom_interaction_table = np.array([
        [-1.09908462, 0.03605344, 1.0354453, -0.68439392, 0.16205739],
        [0.52110612, -1.46853399, 1.22345083, 1.44403716, -0.77540734],
        [1.45091561, -0.5324493, 1.73014703, 0.26964334, 1.83240487],
        [-0.73334607, 1.72870897, 0.15955448, -1.54338156, -0.84544607],
        [-1.30284659, -0.75676422, 1.82499374, 1.85447871, -0.06901917]
    ])

    particles, interaction_table = generate_particles_and_interactions(custom_interaction_table)

    reset_button_rect = draw_button(win)

    slider_spacing = SLIDER_HEIGHT + SLIDER_PADDING + 20
    sliders = {
        'FORCE_DIFF': {'rect': pygame.Rect(PADDING, HEIGHT - BUTTON_HEIGHT - 2 * slider_spacing, SLIDER_WIDTH, SLIDER_HEIGHT), 'value': FORCE_DIFF, 'min': 0, 'max': 10},
        'AMPLIFIER': {'rect': pygame.Rect(PADDING, HEIGHT - BUTTON_HEIGHT - 1 * slider_spacing, SLIDER_WIDTH, SLIDER_HEIGHT), 'value': AMPLIFIER, 'min': 0, 'max': 10},
        'COLLISION_DAMPING': {'rect': pygame.Rect(PADDING, HEIGHT - BUTTON_HEIGHT - 3 * slider_spacing, SLIDER_WIDTH, SLIDER_HEIGHT), 'value': COLLISION_DAMPING, 'min': 0, 'max': 1},
    }

    dragging_slider = None
    dragging_slider_name = None
    dragging_particle = None
    dragging_particle_index = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if reset_button_rect.collidepoint(mouse_pos):
                    particles, interaction_table = generate_particles_and_interactions(custom_interaction_table)
                else:
                    for name, slider in sliders.items():
                        if slider['rect'].collidepoint(mouse_pos):
                            dragging_slider = slider['rect']
                            dragging_slider_name = name
                            break
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = None
                dragging_slider_name = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging_slider:
                    mouse_x, _ = event.pos
                    new_value = (mouse_x - dragging_slider.x) / dragging_slider.width * (sliders[dragging_slider_name]['max'] - sliders[dragging_slider_name]['min']) + sliders[dragging_slider_name]['min']
                    new_value = max(sliders[dragging_slider_name]['min'], min(sliders[dragging_slider_name]['max'], new_value))
                    sliders[dragging_slider_name]['value'] = new_value
                    if dragging_slider_name == 'FORCE_DIFF':
                        FORCE_DIFF = new_value
                    elif dragging_slider_name == 'AMPLIFIER':
                        AMPLIFIER = new_value
                    elif dragging_slider_name == 'COLLISION_DAMPING':
                        COLLISION_DAMPING = new_value

        win.fill((20, 20, 20))

        calculate_forces(particles, interaction_table, AMPLIFIER, COLLISION_DAMPING)
        update_particles(particles)
        draw_particles(particles, win)

        for name, slider in sliders.items():
            draw_slider(win, slider['rect'].x, slider['rect'].y, slider['rect'].width, slider['rect'].height, slider['value'], slider['min'], slider['max'], name)

        reset_button_rect = draw_button(win)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
