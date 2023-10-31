import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

#tiempos
start_time=0
stop_time=0
tiempo=0

# Función Eggholder
def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))


# Función de enfriamiento simulado con seguimiento de convergencia
def simulated_annealing():
    # Parámetros iniciales
    T = 100000  # Temperatura inicial
    T_min = 0.001  # Temperatura mínima
    alpha = 0.93  # Tasa de enfriamiento

    # Solución inicial aleatoria en el rango de la función Eggholder
    x = np.random.uniform(-512, 512)
    y = np.random.uniform(-512, 512)

    # Valor inicial de la función de costo
    current_cost = eggholder(x, y)

    # Seguimiento de la convergencia
    iterations = []
    start_time=time.time();
    while T > T_min:
        # Generar una solución vecina aleatoria
        x_new = x + np.random.uniform(-5, 5)
        y_new = y + np.random.uniform(-5, 5)

        # Asegurarse de que la nueva solución esté dentro del rango
        x_new = np.clip(x_new, -512, 512)
        y_new = np.clip(y_new, -512, 512)

        # Calcular el costo de la nueva solución
        new_cost = eggholder(x_new, y_new)
        
        

        # Calcular la diferencia de costos entre la nueva y la actual
        delta_cost = new_cost - current_cost
        
        # Si la nueva solución es mejor o se acepta con una cierta probabilidad
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            x = x_new
            y = y_new
            current_cost = new_cost
         
            

        # Registrar la iteración y el mejor costo
        iterations.append(current_cost)

        # Enfriar la temperatura
        T *= alpha
    stop_time=time.time()
    tiempo=stop_time-start_time
    
    return x, y, current_cost, iterations, tiempo
    
    
    
    
# Ejecutar el algoritmo 20 veces
results = []

for _ in range(20):
    tiempo=0
    best_x, best_y, best_cost, iterations, tiempo= simulated_annealing()
    results.append((best_x, best_y, best_cost, iterations, tiempo))

# Encontrar la mejor solución entre las 20 ejecuciones
best_solution = min(results, key=lambda x: x[2])

# Crear un DataFrame a partir de los resultados
data = {
    'Corrida': list(range(1, len(results) + 1)),
    'Mejor x': [result[0] for result in results],
    'Mejor y': [result[1] for result in results],
    'Valor de la función': [result[2] for result in results],
    'Num Iteraciones': [len(result[3]) for result in results],
    'Tiempo Empleado': [result[4] for result in results]
}

df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
df.to_excel('resultados4.xlsx', index=False)


# Imprimir resultados de la mejor solución
print("Mejor solución encontrada:")
print("x =", best_solution[0])
print("y =", best_solution[1])
print("Valor de la función Eggholder:", best_solution[2])
print("tiempo =", best_solution[4])

# Graficar la convergencia de la mejor solución
plt.figure(figsize=(15, 10))
plt.plot(best_solution[3])
plt.xlabel("Iteración")
plt.ylabel("Valor de la Función")
plt.title("Convergencia de la Mejor Solución")
plt.grid(True)
plt.show()

