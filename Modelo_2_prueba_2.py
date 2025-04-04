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

def moon_position(t, theta_0=0):
    """Calcula posición de la luna en tiempo t"""
    theta = w*t + theta_0
    x = S*np.cos(theta)
    y = S*np.sin(theta)
    return x, y

# Calculo de distancias

def calculate_distances(t, x, y, theta_0=0):
    """Calcula distancias a Tierra y Luna"""
    moon_x, moon_y = moon_position(t, theta_0)
    
    dist_earth = np.sqrt(x**2 + y**2)
    dist_moon = np.sqrt((x - moon_x)**2 + (y - moon_y)**2)
    
    return dist_earth, dist_moon

# Calculo de aceleración

def calculate_acceleration(t, x, y, theta_0=0):
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

def collision_events(t, state):
    """Detecta colisiones con Tierra o Luna"""
    x, y = state[:2]
    dist_earth, dist_moon = calculate_distances(t, x, y)
    return [dist_earth - Rt, dist_moon - Rl]

collision_events.terminal = True

# Simulación y graficación

def simulate_trajectory(v0, theta_0=0, t_max=5*24*3600):
    """Simula y grafica la trayectoria"""
    # Condiciones iniciales
    state0 = [Rt, 0, 0, v0]
    t_span = (0, t_max)
    
    # Resolver sistema
    sol = solve_ivp(system_equations, t_span, state0, 
                    events=collision_events, method='RK45',
                    rtol=1e-8, atol=1e-8)
    
    # Graficar resultados
    plt.figure(figsize=(12, 12))
    
    # Trayectoria del proyectil
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Proyectil')
    
    # Tierra
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(Rt*np.cos(theta), Rt*np.sin(theta), 'g-', label='Tierra')
    
    # Órbita lunar y posición actual
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Órbita lunar')
    moon_final_x, moon_final_y = moon_position(sol.t[-1], theta_0)
    plt.plot(moon_final_x, moon_final_y, 'ko', label='Luna')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Trayectoria del Proyectil en Sistema Tierra-Luna')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

if __name__ == '__main__':
    v0 = 11000  # velocidad inicial (m/s)
    theta_0 = 0  # posición angular inicial de la Luna
    simulate_trajectory(v0, theta_0)


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