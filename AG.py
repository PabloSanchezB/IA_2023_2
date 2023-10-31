import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd

# Función Egg Holder
def egg_holder(x, y):
    return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))

# Función de inicialización de la población
def initialize_population(pop_size, num_variables):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(-512, 512) for _ in range(num_variables)]
        population.append(individual)
    return population

# Función para evaluar la aptitud de un individuo
def evaluate_fitness(individual):
    x, y = individual
    return egg_holder(x, y)

# Función para seleccionar padres mediante torneo
def select_parents(population, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, 3)
        tournament.sort(key=lambda x: evaluate_fitness(x))
        selected_parents.append(tournament[0])
    return selected_parents

# Función para realizar cruces de dos padres para obtener un hijo
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Función para realizar mutaciones en un individuo
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-512, 512)
    return individual

# Función principal del algoritmo genético
def genetic_algorithm(pop_size, num_generations, mutation_rate, elite_size, crossover_rate):
    num_variables = 2
    population = initialize_population(pop_size, num_variables)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for generation in range(num_generations):
        parents = select_parents(population, pop_size // 2)
        new_population = []
        
        # Ordenar la población actual por aptitud
        population.sort(key=lambda x: evaluate_fitness(x))
        
        # Conservar la élite
        elite = population[:elite_size]
        
        for i in range(0, pop_size - elite_size, 2):
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        # Agregar la élite a la nueva población
        new_population.extend(elite)
        
        population = new_population

        for individual in population:
            fitness = evaluate_fitness(individual)
            if fitness < best_fitness:
                best_solution = individual
                best_fitness = fitness

        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history

resultados=[]
best_sol=[]
if __name__ == "__main__":
    
    for _ in range(20):
        start_time = time.time()
        best_solution, best_fitness, fitness_history = genetic_algorithm(
            pop_size=200, num_generations=200, mutation_rate=0.03, elite_size=15, crossover_rate=0.09
        )
        end_time = time.time()
        resultados.append((best_solution,best_fitness,end_time-start_time,fitness_history))
    best_sol = min(resultados, key=lambda x: x[1])
    
    # Crear un DataFrame a partir de los resultados
    data = {
        'Corrida': list(range(1, len(resultados) + 1)),
        'Mejor (x,y)': [result[0] for result in resultados],
        'Valor de la función': [result[1] for result in resultados],
        'Tiempo Empleado': [result[2] for result in resultados]
    }

    df = pd.DataFrame(data)

    # Guardar el DataFrame en un archivo Excel
    df.to_excel('resultados2.xlsx', index=False)
    print(f"Tiempo empleado: {best_sol[2]} segundos")
    print(f"Número de iteraciones: {len(fitness_history)}")
    print(f"Mejor solución encontrada: {best_solution}")
    print(f"Mejor valor de la función objetivo: {best_fitness}")
    
    plt.figure(figsize=(15, 10))
    plt.plot(best_sol[3])
    plt.xlabel("Generación")
    plt.ylabel("Valor de la función objetivo")
    plt.title("Convergencia del Algoritmo Genético")
    plt.show()
