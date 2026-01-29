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

def equations_of_motion(t, state):
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

def plot_multiple_trajectories(v0_range):
    """Plot trajectories for different initial velocities"""
    plt.figure(figsize=(15, 15))
    
    # Plot Earth
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(Rt*np.cos(theta), Rt*np.sin(theta), 'b-', label='Tierra', linewidth=2)
    
    # Plot Moon's orbit
    plt.plot(S*np.cos(theta), S*np.sin(theta), 'k--', label='Órbita lunar', alpha=0.5)
    
    # Plot Moon at initial position
    plt.plot(S + Rl*np.cos(theta), Rl*np.sin(theta), 'gray', label='Luna')
    
    # Calculate and plot trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, len(v0_range)))
    
    for v0, color in zip(v0_range, colors):
        # Initial conditions
        state0 = [Rt, 0, v0, 0]
        t_span = (0, 5*24*3600)  # 5 days
        
        # Solve the system
        sol = solve_ivp(equations_of_motion, t_span, state0, method='RK45',
                       rtol=1e-8, atol=1e-8)
        
        # Plot trajectory
        plt.plot(sol.y[0], sol.y[1], '-', color=color, alpha=0.6,
                label=f'v₀ = {v0/1000:.1f} km/s')
    
    # Configure plot
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Trayectorias Tierra-Luna para diferentes velocidades iniciales')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Set x and y limits to show only the right side
    plt.xlim(0, S*1.1)  # From x=0 to 110% of Earth-Moon distance
    plt.ylim(-S*0.55, S*0.55)  # Symmetric y limits for proper scaling
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.show()