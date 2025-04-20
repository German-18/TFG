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

# Ecuaciones de movimiento

def equations_of_motion(t, state):
    """
    State vector: [x, y, vx, vy]
    Returns derivatives: [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    
    # Distance calculations
    r_earth = np.sqrt(x**2 + y**2)
    r_moon = np.sqrt((x - S*np.cos(w*t))**2 + (y - S*np.sin(w*t))**2)
    
    # Gravitational accelerations
    ax_earth = -G*Mt*x/(r_earth**3)
    ay_earth = -G*Mt*y/(r_earth**3)
    
    ax_moon = -G*Ml*(x - S*np.cos(w*t))/(r_moon**3)
    ay_moon = -G*Ml*(y - S*np.sin(w*t))/(r_moon**3)
    
    return [vx, vy, ax_earth + ax_moon, ay_earth + ay_moon]

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

# Definición de la función principal

def main(v0):
    # Initial conditions
    x0 = Rt  # Start from Earth's surface
    y0 = 0
    vx0 = v0
    vy0 = 0
    
    state0 = [x0, y0, vx0, vy0]
    t_span = (0, 100*24*3600)  # 100 days in seconds
    
    # Solve the system
    sol = solve_ivp(equations_of_motion, t_span, state0, 
                    events=[collision_moon,collision_earth], method='RK45',
                    rtol=1e-8, atol=1e-8)
    
    print(sol.t[-1])
    
    # Plot trajectory
    plt.figure(figsize=(12, 12))
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Proyectil')
    
    # Plot Earth
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(Rt*np.cos(theta), Rt*np.sin(theta), 'g-', label='Tierra')
    
    # Plot Moon's orbit
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Orbita de la Luna')
    
    # Plot lines to initial and final Moon positions
    plt.plot([0, S], [0, 0], 'r--', label='Posición inicial de la Luna')
    plt.plot([0, S*np.cos(w*sol.t[-1])], [0, S*np.sin(w*sol.t[-1])], 'b--', label='Posición final de la Luna')
    
    # Plot Moon surface only at final position
    plt.plot(S*np.cos(w*sol.t[-1]) + Rl*np.cos(theta), S*np.sin(w*sol.t[-1]) + Rl*np.sin(theta), 'gray', label='Luna')
    
    plt.xlim(0, sol.y[0].max()*1.1)

    # plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Trayectoria del proyectil en el sistema Tierra-Luna')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

#Ejecución del programa

if __name__ == '__main__':
    # Example initial velocity
    v0 = 1.0085*11000  # m/s
    main(v0)

def analyze_combined_parameters(v0_range):
    """
    Analyzes and plots both final velocity and minimum distance to Moon's surface
    for different initial velocities in the same graph
    """
    final_velocities = []
    min_distances = []
    velocities = []
    
    for v0 in v0_range:
        # Initial conditions
        x0 = Rt
        y0 = 0
        vx0 = v0
        vy0 = 0
        
        state0 = [x0, y0, vx0, vy0]
        t_span = (0, 365*24*3600)
        
        # Solve the system
        sol = solve_ivp(equations_of_motion, t_span, state0, 
                       method='RK45', rtol=1e-8, atol=1e-8)
        
        # Calculate final velocity
        final_vx = sol.y[2][-1]
        final_vy = sol.y[3][-1]
        final_v = np.sqrt(final_vx**2 + final_vy**2)
        
        # Calculate minimum distance to Moon
        min_dist = float('inf')
        for i in range(len(sol.t)):
            moon_x = S*np.cos(w*sol.t[i])
            moon_y = S*np.sin(w*sol.t[i])
            dist = np.sqrt((sol.y[0][i] - moon_x)**2 + (sol.y[1][i] - moon_y)**2) - Rl
            min_dist = min(min_dist, dist)
        
        final_velocities.append(final_v)
        min_distances.append(min_dist)
        velocities.append(v0)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot final velocities on left axis (red)
    line1 = ax1.plot(velocities, final_velocities, 'r-', label='Velocidad Final')
    ax1.set_xlabel('Velocidad Inicial (m/s)')
    ax1.set_ylabel('Velocidad Final (m/s)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    # Plot minimum distances on right axis (blue)
    line2 = ax2.plot(velocities, min_distances, 'b-', label='Distancia Mínima')
    ax2.set_ylabel('Distancia Mínima a la Luna (m)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Velocidad Final y Distancia Mínima vs Velocidad Inicial')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == '__main__':
    v0_range = np.linspace(10500, 11500, 50)
    analyze_combined_parameters(v0_range)

# This improved model includes:

# 2D Motion: Models motion in the x-y plane

# Rotating Moon: Includes Moon's orbital motion

# More Accurate Physics:
# Full vector gravitational forces
# Both bodies' gravitational effects at all times

# Better Visualization:
# Plots actual trajectory in 2D
# Shows Earth, Moon orbit, and spacecraft path

# Improved Numerical Integration:
# Uses RK45 method
# Better tolerance settings
# Collision detection for both bodies

# You can run this code with different initial velocities to simulate various trajectories in the Earth-Moon system.

#En funcion de la velocidad inicial, representar la velo