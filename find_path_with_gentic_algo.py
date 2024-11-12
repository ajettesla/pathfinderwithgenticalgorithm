import random
import numpy as np

# Initialize a larger maze as a 2D array (0 = space, 1 = wall, 2 = start, 3 = end)
maze = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
    [1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,3,1,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]
# Possible moves
MOVES = {
    0: (-1, 0),  # left
    1: (1, 0),   # right
    2: (0, 1),   # down
    3: (0, -1)   # up
}

# Calculate population size and generations based on maze size
def calculate_parameters(maze):
    height = len(maze)
    width = len(maze[0])
    population_size = int(width * height * 10)
    generations = int((width * height) / 2)
    move_length = width * height
    return population_size, generations, move_length

# Fitness function to evaluate the distance from start to end and avoid loops
def fitness_function(solution, start, end, maze):
    row, col = start
    visited = set()
    path = [(row, col)]  # Track the path
    fitness = 0
    visited.add((row, col))

    for move in solution:
        dr, dc = MOVES[move]
        new_row, new_col = row + dr, col + dc

        # Check if move is within bounds and not into a wall
        if 0 <= new_row < len(maze) and 0 <= new_col < len(maze[0]) and maze[new_row][new_col] != 1:
            if (new_row, new_col) not in visited:  # Avoid loops
                visited.add((new_row, new_col))
                path.append((new_row, new_col))
                row, col = new_row, new_col

                # Calculate Manhattan distance to the goal
                distance = abs(end[0] - row) + abs(end[1] - col)
                fitness += 1 / (distance + 1)  # Higher fitness for getting closer to the end
            else:
                fitness -= 1  # Penalty for revisiting a point (looping)
        else:
            fitness -= 2  # Penalty for hitting a wall or going out of bounds

        if (row, col) == end:  # Goal reached
            return float('inf'), path

    return max(fitness, 0.1), path  # Ensure a minimum positive fitness

# Crossover between two parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Mutate by flipping a bit in a random move
def mutate(solution, mutation_rate=0.1):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.randint(0, 3)  # Randomly mutate move

# Print matrix with routes and arrows
def print_matrix_with_routes_and_arrows(matrix, path):
    matrix_copy = np.array(matrix, dtype=str)

    def get_direction_arrow(start, end):
        if start[0] == end[0]:
            return '→' if start[1] < end[1] else '←'
        elif start[1] == end[1]:
            return '↓' if start[0] < end[0] else '↑'
        return ' '

    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == 2:
                matrix_copy[r][c] = 'S'
            elif matrix[r][c] == 3:
                matrix_copy[r][c] = 'E'

    if path:
        for i in range(len(path) - 1):
            r, c = path[i]
            next_r, next_c = path[i + 1]
            if matrix_copy[r][c] not in ('S', 'E'):
                matrix_copy[r][c] = get_direction_arrow((r, c), (next_r, next_c))

    for row in matrix_copy:
        print(" ".join(row))
    print("\nCoordinates of Path:")
    print(path)

# Genetic algorithm to solve the maze
def genetic_algorithm(maze, start, end):
    population_size, generations, move_length = calculate_parameters(maze)
    population = [random.choices(range(4), k=move_length) for _ in range(population_size)]

    for generation in range(generations):
        fitness_and_paths = [fitness_function(sol, start, end, maze) for sol in population]
        fitness_scores = [fit for fit, _ in fitness_and_paths]
        paths = [path for _, path in fitness_and_paths]

        if max(fitness_scores) == float('inf'):
            best_path = paths[fitness_scores.index(float('inf'))]
            print(f"Found solution at generation {generation}:")
            print_matrix_with_routes_and_arrows(maze, best_path)
            return best_path

        # Select parents based on fitness
        selected_population = random.choices(population, weights=fitness_scores, k=population_size)

        # Crossover and mutate to create new population
        new_population = []
        for i in range(population_size // 2):
            parent1, parent2 = random.sample(selected_population, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([child1, child2])

        # Apply mutation
        for i in range(population_size):
            mutate(new_population[i])

        population = new_population

    print("No solution found.")
    return None

# Locate start and end in the maze
start = end = None
for row in range(len(maze)):
    for col in range(len(maze[0])):
        if maze[row][col] == 2:
            start = (row, col)
        elif maze[row][col] == 3:
            end = (row, col)

if start and end:
    solution = genetic_algorithm(maze, start, end)
else:
    print("Start or end point missing in the maze.")
