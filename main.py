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
SAFETY_MARGIN = 2
DAMPING = 0.98
COLLISION_DAMPING = 0.5
INTERACTION_RADIUS = 150

NUM_PARTICLE_TYPES = 5
NUM_PARTICLES = 500

CLOCK = 1500

def generate_particles_and_interactions():
    particle_types = {}
    for i in range(NUM_PARTICLE_TYPES):
        particle_type = chr(ord('A') + i)
        color = (
            random.randint(20, 200),  # R
            random.randint(20, 200),  # G
            random.randint(20, 200)   # B
        )
        particle_types[particle_type] = {'color': color}

    interaction_table = np.zeros(
        (NUM_PARTICLE_TYPES, NUM_PARTICLE_TYPES), dtype=np.float64)
    for i in range(NUM_PARTICLE_TYPES):
        for j in range(NUM_PARTICLE_TYPES):
            interaction_table[i, j] = random.uniform(
                FORCE_DIFF * (-1), FORCE_DIFF)

    particles = np.zeros((NUM_PARTICLES, 6), dtype=np.float64)
    for i in range(NUM_PARTICLES):
        ptype = random.choice(list(particle_types.keys()))
        ptype_index = ord(ptype) - ord('A')
        x = random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        y = random.randint(PARTICLE_RADIUS, HEIGHT -
                           PARTICLE_RADIUS - BUTTON_HEIGHT - PADDING)
        color = particle_types[ptype]['color']
        particles[i] = [x, y, 0, 0, ptype_index,
                        color[0] * 65536 + color[1] * 256 + color[2]]
        
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

            if distance < 2 * (PARTICLE_RADIUS + SAFETY_MARGIN):
                overlap = 2 * (PARTICLE_RADIUS + SAFETY_MARGIN) - distance
                if distance > 0:
                    overlap_fraction = overlap / distance
                    fx += collision_damping * overlap_fraction * (dx / distance)
                    fy += collision_damping * overlap_fraction * (dy / distance)
                    particles[i, 2] -= fx
                    particles[i, 3] -= fy
                    particles[j, 2] += fx
                    particles[j, 3] += fy

        particles[i, 2] += fx
        particles[i, 3] += fy


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

    particles, interaction_table = generate_particles_and_interactions()

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
                if reset_button_rect.collidepoint(event.pos):
                    particles, interaction_table = generate_particles_and_interactions()
                else:
                    for slider_name, slider in sliders.items():
                        if slider['rect'].collidepoint(event.pos):
                            dragging_slider = slider
                            dragging_slider_name = slider_name
                            break

                    for i, p in enumerate(particles):
                        x, y, _, _, _, _ = p
                        if np.hypot(event.pos[0] - x, event.pos[1] - y) <= PARTICLE_RADIUS:
                            dragging_particle = p
                            dragging_particle_index = i
                            break

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = None
                dragging_particle = None
                dragging_particle_index = None

            elif event.type == pygame.MOUSEMOTION:
                if dragging_slider is not None:
                    x, y = event.pos
                    slider = dragging_slider
                    min_val, max_val = slider['min'], slider['max']
                    slider_width = slider['rect'].width
                    value = min(max_val, max(min_val, (x - slider['rect'].x) * (max_val - min_val) / slider_width))
                    sliders[dragging_slider_name]['value'] = value
                    if dragging_slider_name == 'FORCE_DIFF':
                        FORCE_DIFF = value
                    elif dragging_slider_name == 'AMPLIFIER':
                        AMPLIFIER = value
                    elif dragging_slider_name == 'COLLISION_DAMPING':
                        COLLISION_DAMPING = value
                
                if dragging_particle is not None:
                    x, y = event.pos
                    particles[dragging_particle_index, 0] = x
                    particles[dragging_particle_index, 1] = y

        win.fill((20, 20, 20))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(calculate_forces, particles, interaction_table, AMPLIFIER, COLLISION_DAMPING)]
            for future in futures:
                future.result()

        update_particles(particles)

        draw_particles(particles, win)

        draw_button(win)

        for slider_name, slider in sliders.items():
            draw_slider(win, slider['rect'].x, slider['rect'].y, slider['rect'].width, slider['rect'].height, slider['value'], slider['min'], slider['max'], slider_name)

        pygame.display.flip()

        pygame.time.Clock().tick(CLOCK)

    pygame.quit()

if __name__ == "__main__":
    main()
