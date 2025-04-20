import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants (SI units)
G = 6.67430e-11     # Gravitational constant
Mt = 5.972e24       # Earth mass
Ml = 7.34767309e22  # Moon mass
Rt = 6.371e6        # Earth radius
Rl = 1.737e6        # Moon radius
S = 3.844e8         # distancia entre los centros de la tierra y la luna.
w = 2.662e-6        # Moon's angular velocity

# Posicion de la Luna

def moon_position(t, theta_0=-np.pi/4): #Poner theta_0 para que la luna empiece en un punto diferente¡¡¡¡¡¡¡¡¡
    """Calcula posición de la luna en tiempo t"""
    theta = w*t + theta_0
    x = S*np.cos(theta)
    y = S*np.sin(theta)
    return x, y

# Calculo de distancias

def calculate_distances(t, x, y, theta_0=-np.pi/4):#Poner theta_0 para que la luna empiece en un punto diferente¡¡¡¡¡¡¡¡¡
    """Calcula distancias a Tierra y Luna"""
    moon_x, moon_y = moon_position(t, theta_0)
    
    dist_earth = np.sqrt(x**2 + y**2)
    dist_moon = np.sqrt((x - moon_x)**2 + (y - moon_y)**2)
    
    return dist_earth, dist_moon

# Calculo de aceleración

def calculate_acceleration(t, x, y, theta_0=-np.pi/4): #Poner theta_0 para que la luna empiece en un punto diferente¡¡¡¡¡¡¡¡¡
    """Calcula aceleración del proyectil"""
    moon_x, moon_y = moon_position(t, theta_0)
    dist_earth, dist_moon = calculate_distances(t, x, y, theta_0)
    
    # Aceleración debido a la Tierra
    ax_earth = -G*Mt*x/(dist_earth**3)
    ay_earth = -G*Mt*y/(dist_earth**3)
    
    # Aceleración debido a la Luna
    ax_moon = -G*Ml*(x - moon_x)/(dist_moon**3)
    ay_moon = -G*Ml*(y - moon_y)/(dist_moon**3)
    
    return ax_earth + ax_moon, ay_earth + ay_moon

# Definición del sistema de ecuaciones diferenciales

def system_equations(t, state):
    """Sistema de ecuaciones diferenciales"""
    x, y, vx, vy = state
    ax, ay = calculate_acceleration(t, x, y)
    return [vx, vy, ax, ay]

# Definición de eventos para detectar colisiones

# Definición de función eventos que detecta colision con la Luna
def collision_moon(t, state):
    x, y = state[:2]
    r_moon = np.sqrt((x - S*np.cos(w*t))**2 + (y - S*np.sin(w*t))**2)
    disSupMoon = S-r_moon-Rl #Distancia entre el satélite y la Luna
    return disSupMoon
collision_moon.terminal = True # La integración se detiene cuando la función de eventos es cero

# Definición de función eventos que detecta colision con la Tierra
def collision_earth(t, state):
    x, y = state[:2]
    r_earth = np.sqrt(x**2 + y**2)
    disSupEarth = r_earth-(Rt-10) #Distancia entre el satélite y 10 metros por debajo de la superficie de la Tierra
    return disSupEarth
collision_earth.terminal = True # La integración se detiene cuando la función de eventos es cero

# Simulación y graficación

def simulate_trajectory(v0, theta_0=-np.pi/4, t_max=100*24*3600):
    """Simula y grafica la trayectoria"""
    # Condiciones iniciales
    state0 = [Rt, 0, v0, 0]
    t_span = (0, t_max)
    
    # Resolver sistema
    sol = solve_ivp(system_equations, t_span, state0, 
                    events=[collision_moon,collision_earth], method='RK45',
                    rtol=1e-8, atol=1e-8)
    
    # Configurar el plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Trayectoria del proyectil
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Trayectoria del proyectil')
    
    # Órbita lunar (círculo de radio S)
    theta = np.linspace(0, 2*np.pi, 100)
    luna_x = S*np.cos(theta)
    luna_y = S*np.sin(theta)
    plt.plot(luna_x, luna_y, 'k--', label='Órbita lunar')
    
    # Tierra a escala (círculo sólido)
    tierra = plt.Circle((0, 0), Rt, color='blue', fill=True, alpha=0.3, label='Tierra')
    ax.add_patch(tierra)
    
    # Luna inicial a escala
    moon_init_x, moon_init_y = moon_position(0, theta_0)
    luna_inicial = plt.Circle((moon_init_x, moon_init_y), Rl, color='lightgray', 
                            fill=True, alpha=0.3, label='Luna (posición inicial)')
    ax.add_patch(luna_inicial)
    
    # Luna final a escala
    moon_final_x, moon_final_y = moon_position(sol.t[-1], theta_0)
    luna_final = plt.Circle((moon_final_x, moon_final_y), Rl, color='gray', 
                           fill=True, alpha=0.5, label='Luna (posición final)')
    ax.add_patch(luna_final)
    
    # Líneas de posición inicial y final de la Luna
    plt.plot([0, moon_init_x], [0, moon_init_y], 'r--', label='Posición inicial Luna')
    plt.plot([0, moon_final_x], [0, moon_final_y], 'b--', label='Posición final Luna')
    
    # Configuración del plot
    ax.set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Sistema Tierra-Luna-Proyectil (Tamaños a escala)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Ajustar límites para ver todo el sistema
    limit = S * 1.1  # 10% más grande que la distancia Tierra-Luna
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    
    plt.show()

if __name__ == '__main__':
    v0 = 11184  # velocidad inicial (m/s)
    theta_0 = -np.pi/4  # posición angular inicial de la Luna
    simulate_trajectory(v0, theta_0)

def analyze_min_distance(v0_range, theta_0 = -np.pi/4):
    """Analiza la distancia mínima a la superficie lunar para diferentes velocidades iniciales"""
    min_distances = []
    velocities = []
    
    for v0 in v0_range:
        # Condiciones iniciales
        state0 = [Rt, 0, v0, 0]
        t_span = (0, 5*24*3600)
        
        # Resolver sistema
        sol = solve_ivp(system_equations, t_span, state0, 
                       method='RK45', rtol=1e-8, atol=1e-8)
        
        # Calcular distancia mínima a la Luna
        min_dist = float('inf')
        for i in range(len(sol.t)):
            moon_x, moon_y = moon_position(sol.t[i], theta_0)
            dist = np.sqrt((sol.y[0][i] - moon_x)**2 + (sol.y[1][i] - moon_y)**2) - Rl
            min_dist = min(min_dist, dist)
        
        min_distances.append(min_dist)
        velocities.append(v0)
    
    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, min_distances, 'b-')
    plt.grid(True)
    plt.title('Distancia Mínima a la Superficie Lunar vs Velocidad Inicial')
    plt.xlabel('Velocidad Inicial (m/s)')
    plt.ylabel('Distancia Mínima (m)')
    plt.show()

# Ejemplo de uso
if __name__ == '__main__':
    # Probar rango de velocidades de 10.5 km/s a 11.5 km/s
    v0_range = np.linspace(10500, 11500, 50)
    analyze_min_distance(v0_range)

def analyze_min_distance_3d(v0_range, theta_range):
    """
    Analiza y visualiza en 3D la distancia mínima a la superficie lunar para diferentes 
    velocidades iniciales y posiciones iniciales de la Luna
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Crear mallas 2D para velocidades y ángulos
    V, T = np.meshgrid(v0_range, theta_range)
    min_distances = np.zeros_like(V)
    
    # Contador de progreso
    total = len(v0_range) * len(theta_range)
    counter = 0
    
    for i, theta in enumerate(theta_range):
        for j, v0 in enumerate(v0_range):
            # Condiciones iniciales
            state0 = [Rt, 0, v0, 0]
            t_span = (0, 5*24*3600)
            
            # Resolver sistema
            sol = solve_ivp(lambda t, state: system_equations(t, state, theta), 
                          t_span, state0, method='RK45', rtol=1e-8, atol=1e-8)
            
            # Calcular distancia mínima a la Luna
            min_dist = float('inf')
            for k in range(len(sol.t)):
                moon_x, moon_y = moon_position(sol.t[k], theta)
                dist = np.sqrt((sol.y[0][k] - moon_x)**2 + 
                             (sol.y[1][k] - moon_y)**2) - Rl
                min_dist = min(min_dist, dist)
            
            min_distances[i,j] = min_dist
            
            # Actualizar progreso
            counter += 1
            if counter % 10 == 0:
                print(f"Progreso: {counter}/{total}")
    
    # Crear gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar superficie
    surf = ax.plot_surface(V, T*180/np.pi, min_distances, 
                          cmap='viridis', 
                          edgecolor='none')
    
    # Añadir barra de color
    fig.colorbar(surf, ax=ax, label='Distancia Mínima a la Superficie Lunar (m)')
    
    # Etiquetas y título
    ax.set_xlabel('Velocidad Inicial (m/s)')
    ax.set_ylabel('Posición Inicial Luna (grados)')
    ax.set_zlabel('Distancia Mínima (m)')
    ax.set_title('Vista 3D: Distancia Mínima vs Condiciones Iniciales')
    
    # Rotar vista para mejor visualización
    ax.view_init(elev=30, azim=45)
    
    plt.show()

# Ejemplo de uso
if __name__ == '__main__':
    # Definir rangos de análisis
    v0_range = np.linspace(10500, 11500, 20)  # 20 velocidades diferentes
    theta_range = np.linspace(-np.pi/4, 0, 15)  # 15 ángulos diferentes
    analyze_min_distance_3d(v0_range, theta_range)

# Está organizado por funciones específicas:
# - moon_position: Calcula posición lunar
# - calculate_distances: Evalúa distancias
# - calculate_acceleration: Calcula aceleraciones

# Tiene mejor modularidad:
# - Funciones independientes y reutilizables
# - Parámetros claramente definidos
# - Documentación de cada función

# Incluye mejoras visuales:
# - Muestra posición actual de la Luna
# - Gráficas más informativas
# - Etiquetas en español

# Es más flexible:
# - Permite especificar posición inicial de la Luna
# - Tiempo de simulación ajustable
# - Parámetros de integración configurables
# - Para ejecutar diferentes simulaciones, puedes modificar v0 y theta_0 en la sección if __name__ == '__main__'.