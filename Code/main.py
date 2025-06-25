import pygame
import sys
import math
import random
import numpy as np

# Initialisation
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IA de Course - Algorithme Génétique")

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (40, 40, 60)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

# Paramètres voiture
car_width, car_height = 40, 20
max_speed = 5
acceleration = 0.2
turn_speed = 4

# Murs
walls = [
    pygame.Rect(50, 50, 700, 10),
    pygame.Rect(50, 540, 700, 10),
    pygame.Rect(50, 50, 10, 500),
    pygame.Rect(740, 50, 10, 500),
    pygame.Rect(200, 150, 400, 10),
    pygame.Rect(200, 150, 10, 300),
    pygame.Rect(400, 250, 10, 300),
    pygame.Rect(600, 150, 10, 300),
]

# Checkpoints
checkpoints = [
    (180, 110, 40),  # Départ/arrivée
    (670, 150, 40),
    (600, 495, 40),
    (400, 210, 40),
    (180, 485, 40)
]


class NeuralNetwork:
    def __init__(self, weights=None):
        # Réseau : 5 capteurs + vitesse -> 6 neurones cachés -> 3 sorties (accél, freiner, tourner)
        if weights is None:
            self.weights1 = np.random.randn(6, 8) * 2 - 1  # -1 à 1
            self.weights2 = np.random.randn(8, 3) * 2 - 1
        else:
            self.weights1, self.weights2 = weights

    def forward(self, inputs):
        # Normalisation des entrées
        inputs = np.array(inputs) / 200.0  # Normaliser les distances des capteurs
        inputs = np.clip(inputs, 0, 1)

        # Couche cachée
        hidden = np.tanh(np.dot(inputs, self.weights1))

        # Couche de sortie
        output = np.tanh(np.dot(hidden, self.weights2))

        return output

    def get_weights(self):
        return (self.weights1.copy(), self.weights2.copy())


class Car:
    def __init__(self, brain=None):
        self.x = 100
        self.y = 100
        self.angle = 0
        self.speed = 0
        self.alive = True
        self.fitness = 0
        self.checkpoints_reached = 0
        self.current_checkpoint = 0
        self.distance_traveled = 0
        self.last_x, self.last_y = self.x, self.y
        self.stuck_timer = 0
        self.brain = NeuralNetwork() if brain is None else brain

        # Performance tracking
        self.start_time = pygame.time.get_ticks()
        self.checkpoint_times = []

    def get_sensor_data(self):
        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        distances = []

        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            distances.append(distance)

        # Ajouter la vitesse comme entrée
        distances.append(self.speed / max_speed)

        return distances

    def cast_ray(self, angle):
        ray_x, ray_y = self.x, self.y
        step = 2
        max_distance = 200

        dx = math.cos(math.radians(angle)) * step
        dy = math.sin(math.radians(angle)) * step

        for i in range(int(max_distance / step)):
            ray_x += dx
            ray_y += dy

            if ray_x < 0 or ray_x >= WIDTH or ray_y < 0 or ray_y >= HEIGHT:
                return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

            for wall in walls:
                if wall.collidepoint(ray_x, ray_y):
                    return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

        return max_distance

    def update(self):
        if not self.alive:
            return

        # Obtenir les données des capteurs
        sensor_data = self.get_sensor_data()

        # Décision de l'IA
        outputs = self.brain.forward(sensor_data)

        # Interpréter les sorties
        accelerate = outputs[0] > 0.1
        brake = outputs[1] > 0.1
        turn_left = outputs[2] > 0.3
        turn_right = outputs[2] < -0.3

        # Mise à jour de la vitesse
        if accelerate and not brake:
            self.speed = min(self.speed + acceleration, max_speed)
        elif brake:
            self.speed = max(self.speed - acceleration, -max_speed / 2)
        else:
            self.speed *= 0.95

        # Rotation
        if turn_left:
            self.angle -= turn_speed
        elif turn_right:
            self.angle += turn_speed

        # Mouvement
        old_x, old_y = self.x, self.y
        dx = math.cos(math.radians(self.angle)) * self.speed
        dy = math.sin(math.radians(self.angle)) * self.speed
        new_x = self.x + dx
        new_y = self.y + dy

        # Vérifier les collisions
        if self.check_collision(new_x, new_y):
            self.alive = False
            return

        self.x, self.y = new_x, new_y

        # Calculer la distance parcourue
        self.distance_traveled += math.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)

        # Vérifier les checkpoints
        self.check_checkpoints()

        # Détecter si la voiture est bloquée
        if abs(self.x - self.last_x) < 0.5 and abs(self.y - self.last_y) < 0.5:
            self.stuck_timer += 1
            if self.stuck_timer > 180:  # 3 secondes à 60 FPS
                self.alive = False
        else:
            self.stuck_timer = 0
            self.last_x, self.last_y = self.x, self.y

        # Calculer le fitness
        self.calculate_fitness()

    def check_collision(self, x, y):
        car_rect = pygame.Rect(0, 0, car_width, car_height)
        car_rect.center = (x, y)
        for wall in walls:
            if car_rect.colliderect(wall):
                return True
        return False

    def check_checkpoints(self):
        cx, cy, radius = checkpoints[self.current_checkpoint]
        distance = math.hypot(self.x - cx, self.y - cy)

        if distance < radius:
            if self.current_checkpoint == 0 and self.checkpoints_reached > 0:
                # Tour complet !
                current_time = pygame.time.get_ticks()
                lap_time = current_time - self.start_time
                self.checkpoint_times.append(lap_time)
                self.fitness += 10000  # Bonus énorme pour finir un tour

            self.checkpoints_reached += 1
            self.current_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
            self.fitness += 1000  # Bonus pour atteindre un checkpoint

    def calculate_fitness(self):
        # Fitness basé sur les checkpoints atteints et la distance
        checkpoint_bonus = self.checkpoints_reached * 1000
        distance_bonus = self.distance_traveled * 0.1

        # Bonus pour la progression vers le prochain checkpoint
        if self.current_checkpoint < len(checkpoints):
            cx, cy, _ = checkpoints[self.current_checkpoint]
            distance_to_checkpoint = math.hypot(self.x - cx, self.y - cy)
            proximity_bonus = max(0, 200 - distance_to_checkpoint)
        else:
            proximity_bonus = 0

        # Bonus de temps (survie)
        time_bonus = (pygame.time.get_ticks() - self.start_time) * 0.01

        self.fitness = checkpoint_bonus + distance_bonus + proximity_bonus + time_bonus

    def draw(self, color=RED):
        if not self.alive:
            color = (100, 100, 100)  # Gris pour les voitures mortes

        car_surface = pygame.Surface((car_width, car_height))
        car_surface.fill(color)
        car_surface.set_colorkey(BLACK)
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, rect.topleft)

    def draw_sensors(self):
        if not self.alive:
            return

        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            end_x = self.x + math.cos(math.radians(self.angle + sensor_angle)) * distance
            end_y = self.y + math.sin(math.radians(self.angle + sensor_angle)) * distance
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)


class GeneticAlgorithm:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.generation = 1
        self.cars = [Car() for _ in range(population_size)]
        self.best_fitness = 0
        self.best_car = None

    def selection(self):
        # Sélection par tournoi
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.cars, 5)
            winner = max(tournament, key=lambda car: car.fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        # Croisement uniforme
        w1_p1, w2_p1 = parent1.brain.get_weights()
        w1_p2, w2_p2 = parent2.brain.get_weights()

        # Masque de croisement
        mask1 = np.random.random(w1_p1.shape) < 0.5
        mask2 = np.random.random(w2_p1.shape) < 0.5

        w1_child = np.where(mask1, w1_p1, w1_p2)
        w2_child = np.where(mask2, w2_p1, w2_p2)

        return NeuralNetwork((w1_child, w2_child))

    def mutate(self, brain, mutation_rate=0.1, mutation_strength=0.5):
        w1, w2 = brain.get_weights()

        # Mutation des poids
        if random.random() < mutation_rate:
            w1 += np.random.normal(0, mutation_strength, w1.shape)
        if random.random() < mutation_rate:
            w2 += np.random.normal(0, mutation_strength, w2.shape)

        return NeuralNetwork((w1, w2))

    def evolve(self):
        # Trier par fitness
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

        # Garder les statistiques
        self.best_fitness = self.cars[0].fitness
        self.best_car = self.cars[0]

        print(f"Génération {self.generation}: Meilleur fitness = {self.best_fitness:.1f}, "
              f"Checkpoints = {self.best_car.checkpoints_reached}")

        # Nouvelle génération
        new_cars = []

        # Garder les 10% meilleurs (élitisme)
        elite_count = max(1, self.population_size // 10)
        for i in range(elite_count):
            new_cars.append(Car(self.cars[i].brain))

        # Créer le reste par croisement et mutation
        parents = self.selection()
        while len(new_cars) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child_brain = self.crossover(parent1, parent2)
            child_brain = self.mutate(child_brain)
            new_cars.append(Car(child_brain))

        self.cars = new_cars
        self.generation += 1


def draw_checkpoints():
    font = pygame.font.Font(None, 24)
    for i, (x, y, radius) in enumerate(checkpoints):
        pygame.draw.circle(screen, BLUE, (x, y), radius, 3)
        label = "Départ" if i == 0 else f"{i}"
        text = font.render(label, True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)


def main():
    clock = pygame.time.Clock()
    ga = GeneticAlgorithm(population_size=30)
    simulation_speed = 1
    show_sensors = True

    generation_timer = pygame.time.get_ticks()
    max_generation_time = 30000  # 30 secondes par génération

    font = pygame.font.Font(None, 24)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_speed = 3 if simulation_speed == 1 else 1
                elif event.key == pygame.K_r:
                    ga.evolve()
                    generation_timer = pygame.time.get_ticks()
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors

        # Mise à jour des voitures
        for _ in range(simulation_speed):
            alive_cars = 0
            for car in ga.cars:
                if car.alive:
                    car.update()
                    alive_cars += 1

            # Nouvelle génération si toutes les voitures sont mortes ou temps écoulé
            current_time = pygame.time.get_ticks()
            if alive_cars == 0 or (current_time - generation_timer) > max_generation_time:
                ga.evolve()
                generation_timer = current_time
                break

        # Affichage
        screen.fill(GRAY)

        # Dessiner le circuit
        for wall in walls:
            pygame.draw.rect(screen, WHITE, wall)

        draw_checkpoints()

        # Dessiner les voitures
        best_car = max(ga.cars, key=lambda car: car.fitness)
        for i, car in enumerate(ga.cars):
            color = GREEN if car == best_car else RED
            car.draw(color)
            if show_sensors and car == best_car:
                car.draw_sensors()

        # Interface
        gen_text = font.render(f"Génération: {ga.generation}", True, WHITE)
        fitness_text = font.render(f"Meilleur fitness: {ga.best_fitness:.1f}", True, WHITE)
        alive_text = font.render(f"Vivantes: {sum(1 for car in ga.cars if car.alive)}", True, WHITE)
        checkpoints_text = font.render(f"Checkpoints: {best_car.checkpoints_reached}", True, WHITE)

        controls_text = font.render("ESPACE: Accélérer | R: Nouvelle génération | S: Capteurs", True, WHITE)

        screen.blit(gen_text, (10, 10))
        screen.blit(fitness_text, (10, 35))
        screen.blit(alive_text, (10, 60))
        screen.blit(checkpoints_text, (10, 85))
        screen.blit(controls_text, (10, HEIGHT - 25))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
