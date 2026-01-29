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

# ...existing code with constants...

# Añadir después de las constantes existentes
SCALE_FACTOR = 5  # Factor para hacer los cuerpos celestes más visibles

def moon_position(t, theta_0):
    """Calculate Moon position at time t with initial angle theta_0"""
    theta = w*t + theta_0
    return S*np.cos(theta), S*np.sin(theta)

def calculate_acceleration(t, x, y, theta_0):
    """Calculate acceleration components due to Earth and Moon gravity"""
    # Earth's gravitational acceleration
    r_earth = np.sqrt(x**2 + y**2)
    ax_earth = -G*Mt*x/(r_earth**3)
    ay_earth = -G*Mt*y/(r_earth**3)
    
    # Moon's position and gravitational acceleration
    moon_x, moon_y = moon_position(t, theta_0)
    r_moon = np.sqrt((x - moon_x)**2 + (y - moon_y)**2)
    ax_moon = -G*Ml*(x - moon_x)/(r_moon**3)
    ay_moon = -G*Ml*(y - moon_y)/(r_moon**3)
    
    return ax_earth + ax_moon, ay_earth + ay_moon

def system_equations(t, state):
    """System of differential equations"""
    x, y, vx, vy = state
    ax, ay = calculate_acceleration(t, x, y, -np.pi/4)  # Fixed Moon initial position
    return [vx, vy, ax, ay]

# ...rest of existing code...

def plot_velocity_variations(v0_range, theta_0=-np.pi/4):
    """Plot trajectories for different initial velocities with fixed Moon position"""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
 # Plot Earth with increased visual size
    earth_circle = plt.Circle((0, 0), Rt*SCALE_FACTOR, color='blue', 
                            fill=True, alpha=0.3, label='Tierra')
    ax.add_patch(earth_circle)
    
    # Plot Moon's orbit
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Órbita lunar', alpha=0.5)
    
    # Plot Moon with increased visual size
    moon_x, moon_y = moon_position(0, theta_0)
    moon_circle = plt.Circle((moon_x, moon_y), Rl*SCALE_FACTOR, 
                           color='gray', fill=True, alpha=0.5, label='Luna')
    ax.add_patch(moon_circle)
    
    # Add line connecting Earth and Moon
    plt.plot([0, moon_x], [0, moon_y], 'k:', alpha=0.5, linewidth=1)
    
    # Calculate and plot trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, len(v0_range)))
    for v0, color in zip(v0_range, colors):
        state0 = [Rt, 0, v0, 0]
        t_span = (0, 5*24*3600)  # 5 days
        
        sol = solve_ivp(system_equations, t_span, state0, method='RK45',
                       rtol=1e-8, atol=1e-8)
        
        plt.plot(sol.y[0], sol.y[1], '-', color=color, alpha=0.6,
                label=f'v₀ = {v0/1000:.1f} km/s')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, S*1.1)
    plt.ylim(-S*0.55, S*0.55)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Trayectorias para diferentes velocidades iniciales\n'
             f'(posición Luna fija θ₀={theta_0*180/np.pi:.0f}°)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_moon_position_variations(theta_range, v0=11000):
    """Plot trajectories for different Moon positions with fixed initial velocity"""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot Earth with increased visual size
    earth_circle = plt.Circle((0, 0), Rt*SCALE_FACTOR, color='blue', 
                            fill=True, alpha=0.3, label='Tierra')
    ax.add_patch(earth_circle)
    
    # Plot Moon's orbit
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Órbita lunar', alpha=0.5)
    
    # Create arrays to store Moon positions for the connecting line
    moon_positions_x = []
    moon_positions_y = []
    
    # Calculate and plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_range)))
    for theta_0, color in zip(theta_range, colors):
        moon_x, moon_y = moon_position(0, theta_0)
        moon_positions_x.append(moon_x)
        moon_positions_y.append(moon_y)
        
        # Plot Moon with increased visual size
        moon_circle = plt.Circle((moon_x, moon_y), Rl*SCALE_FACTOR, 
                               color=color, fill=True, alpha=0.3, 
                               label=f'Luna θ₀={theta_0*180/np.pi:.0f}°')
        ax.add_patch(moon_circle)
        
        # Add line connecting Earth and Moon
        plt.plot([0, moon_x], [0, moon_y], ':', color=color, alpha=0.3, linewidth=1)
        
        state0 = [Rt, 0, v0, 0]
        t_span = (0, 5*24*3600)
        
        def modified_system(t, state):
            x, y, vx, vy = state
            ax, ay = calculate_acceleration(t, x, y, theta_0)
            return [vx, vy, ax, ay]
        
        sol = solve_ivp(modified_system, t_span, state0, method='RK45',
                       rtol=1e-8, atol=1e-8)
        
        plt.plot(sol.y[0], sol.y[1], '-', color=color, alpha=0.6,
                label=f'θ₀ = {theta_0*180/np.pi:.0f}°')
    

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, S*1.1)
    plt.ylim(-S*0.55, S*0.55)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Trayectorias para diferentes posiciones iniciales de la Luna\n'
             f'(velocidad fija v₀={v0/1000:.1f} km/s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Ejemplo 1: Diferentes velocidades con posición lunar fija
    print("Generando gráfico para diferentes velocidades iniciales...")
    v0_range = np.linspace(11000, 11200, 8)  # 8 velocidades entre 11.0 y 11.2 km/s
    theta_0 = -np.pi/4  # Posición fija de la Luna a -45°
    plot_velocity_variations(v0_range, theta_0)
    
    # Pausa entre gráficos
    input("Presione Enter para ver el siguiente gráfico...")
    
    # Ejemplo 2: Diferentes posiciones lunares con velocidad fija
    print("Generando gráfico para diferentes posiciones iniciales de la Luna...")
    theta_range = np.linspace(-np.pi/2, 0, 8)  # 8 posiciones diferentes
    v0 = 11141  # Velocidad fija que permite alcanzar la Luna
    plot_moon_position_variations(theta_range, v0)
