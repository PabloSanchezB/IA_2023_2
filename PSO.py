import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

#tiempos
start_time=0
stop_time=0
tiempo=0

def egg_holder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def pso_minimize(func, n_particles, n_iterations, bounds):
    # Inicialización de partículas
    particles = np.random.uniform(bounds[0], bounds[1], (n_particles, 2))
    velocities = np.random.uniform(-1, 1, (n_particles, 2))
    best_positions = particles.copy()
    best_values = np.array([func(*p) for p in best_positions])
    global_best_idx = np.argmin(best_values)
    global_best = best_positions[global_best_idx]
    
    # Parámetros de PSO
    w = 1.2  # Inercia
    c1 = 0.7  # Aprendizaje cognitivo
    c2 = 3.2  # Aprendizaje social
    
    # Historial de convergencia
    convergence = []
    start_time=time.time()
    for i in range(n_iterations):
        for j in range(n_particles):
            # Actualizar la velocidad y posición de la partícula
            r1, r2 = np.random.rand(2)
            velocities[j] = w * velocities[j] + c1 * r1 * (best_positions[j] - particles[j]) + c2 * r2 * (global_best - particles[j])
            particles[j] = particles[j] + velocities[j]
            
            # Limitar las partículas dentro de los límites
            particles[j] = np.clip(particles[j], bounds[0], bounds[1])
            
            # Evaluar la función objetivo en la nueva posición
            value = func(*particles[j])
            
            # Actualizar la mejor posición de la partícula
            if value < best_values[j]:
                best_values[j] = value
                best_positions[j] = particles[j]
                
                # Actualizar el mejor global si es necesario
                if value < best_values[global_best_idx]:
                    global_best_idx = j
                    global_best = particles[j]
        
        # Registrar la mejor convergencia en esta iteración
        convergence.append(best_values[global_best_idx])
    stop_time=time.time()
    tiempo=stop_time-start_time
    return global_best, best_values[global_best_idx], i+1, convergence, tiempo

# Parámetros
n_particles = 150
n_iterations = 170
bounds = (-512, 512)

# Ejecutar PSO
#best_solution, best_value, num_iterations, convergence, tiempo = pso_minimize(egg_holder, n_particles, n_iterations, bounds)

# Ejecutar el algoritmo 20 veces
results = []

for _ in range(20):
    tiempo=0
    best_sol, best_value, iterations, convergence, tiempo= pso_minimize(egg_holder, n_particles, n_iterations, bounds)
    results.append((best_sol, best_value, iterations, tiempo))

# Encontrar la mejor solución entre las 20 ejecuciones
best_solution = min(results, key=lambda x: x[2])

# Imprimir resultados
print("Mejor solución encontrada:", best_solution[0])
print("Valor de la función objetivo en la mejor solución:", best_solution[1])
print("Número de iteraciones:", best_solution[2])
print("tiempo: ", best_solution[3])

# Crear un DataFrame a partir de los resultados
data = {
    'Corrida': list(range(1, len(results) + 1)),
    'Mejor (x,y)': [result[0] for result in results],
    'Valor de la función': [result[1] for result in results],
    'Num Iteraciones': [result[2] for result in results],
    'Tiempo Empleado': [result[3] for result in results]
}

df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
df.to_excel('resultados3.xlsx', index=False)

# Graficar convergencia
plt.figure(figsize=(15, 10))
plt.plot(convergence)
plt.xlabel("Iteración")
plt.ylabel("Valor de la función objetivo")
plt.title("Convergencia de PSO")
plt.show()
