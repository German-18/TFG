import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants (SI units)
G = 6.67430e-11     # Gravitational constant
Mt = 5.972e24       # Earth mass
Ml = 7.34767309e22  # Moon mass
Rt = 6.371e6        # Earth radius
Rl = 1.737e6        # Moon radius
S = 3.844e8         # Earth-Moon distance
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

# Detección de colisiones

def collision_event(t, state):
    """
    Detect collisions with Earth or Moon
    """
    x, y = state[:2]
    r_earth = np.sqrt(x**2 + y**2)
    r_moon = np.sqrt((x - S*np.cos(w*t))**2 + (y - S*np.sin(w*t))**2)
    
    return [r_earth - Rt, r_moon - Rl]

collision_event.terminal = True

# Definición de la función principal

def main(v0):
    # Initial conditions
    x0 = Rt  # Start from Earth's surface
    y0 = 0
    vx0 = 0
    vy0 = v0
    
    state0 = [x0, y0, vx0, vy0]
    t_span = (0, 5*24*3600)  # 5 days simulation
    
    # Solve the system
    sol = solve_ivp(equations_of_motion, t_span, state0, 
                    events=collision_event, method='RK45',
                    rtol=1e-8, atol=1e-8)
    
    # Plot trajectory
    plt.figure(figsize=(12, 12))
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Spacecraft')
    
    # Plot Earth
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(Rt*np.cos(theta), Rt*np.sin(theta), 'g-', label='Earth')
    
    # Plot Moon's orbit
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Moon orbit')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Spacecraft Trajectory in Earth-Moon System')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

#Ejecución del programa

if __name__ == '__main__':
    # Example initial velocity
    v0 = 11000  # m/s
    main(v0)

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