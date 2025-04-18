import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
    
# Constants
G = 6.6726e-11
Mt = 5.975e24
Ml = 7.35e22
Rt = 6.378e6
Rl = 1.738e6
S = 3.844e8

# main function
def main(v0):

    # solve journey with initial velocity v0
    sol = solve_journey(v0)

    #plot the results
    plot_journey(sol)

    # define range of initial velocities
    number_of_velocities = 1000
    v0_range = np.linspace(0.9*v0, 1.1*v0, number_of_velocities)

    # solve journey for each initial velocity
    results = []
    for idx in range(len(v0_range)):
        print(f"Calculating for initial velocity {idx} / {number_of_velocities}")
        v = v0_range[idx]
        sol = solve_journey(v)
        results.append(sol.y[1][-1])  # store final velocity at the end of the journey

    # plot the results
    plt.figure(figsize=(10,6))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0
    plt.plot(v0_range, results,'o-')
    plt.xlabel('Initial Velocity (m/s)')
    plt.ylabel('Final Velocity (m/s)')
    plt.title('Final Velocity vs Initial Velocity')
    plt.show()

# Define the system of differential equations
def derivadas(t, y):
    r, v = y
    return [v, aceleracion(t, r)]

def aceleracion(t, r):
    return -G*Mt/(r**2) + G*Ml/(S-r)**2

# Definición de función eventos que detecta colision con la Luna
def collision_moon(t, y):
    disSupMoon = S-y[0]-Rl #Distancia entre el satélite y la Luna
    return disSupMoon
collision_moon.terminal = True # La integración se detiene cuando la función de eventos es cero

# Definición de función eventos que detecta colision con la Tierra
def collision_earth(t, y):
    disSupEarth = y[0]-(Rt-10) #Distancia entre el satélite y 10 metros por debajo de la superficie de la Tierra
    return disSupEarth
collision_earth.terminal = True # La integración se detiene cuando la función de eventos es cero

def solve_journey(v0):
    # Initial conditions
    y0 = [Rt, v0]  # [r(0), v(0)]
    t_span = (0, 1000000)  # Intervalo de tiempo

    # Solve IVP
    sol = solve_ivp(derivadas,t_span,y0,method='RK45',rtol=1e-8,atol=1e-8,events=[collision_moon, collision_earth])
    
    return sol

def plot_journey(sol):
    # Plotting velocity
    plt.figure(figsize=(10,6))
    plt.plot(sol.t, sol.y[1])  # sol.y[0] es r(t), sol.y[1] es v(t)
    plt.legend(['Velocity'], loc='best')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Orbital Velocity vs Time')
    plt.grid(True)

    # Plotting position
    plt.figure(figsize=(10,6))
    plt.plot(sol.t, sol.y[0])  # sol.y[0] es r(t), sol.y[1] es v(t)
    plt.legend(['Position'], loc='best')
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.title('Orbital Position vs Time')
    plt.grid(True)
    
    plt.show()

# Call main function
if __name__ == '__main__':
    
    # calculation of initial velocity
    r1 = S * (np.sqrt(Mt)/(np.sqrt(Mt) + np.sqrt(Ml)))
    v0 = np.sqrt(2) * np.sqrt(G*Mt/Rt - G*Ml/(S-Rt) - G*Mt/r1 + G*Ml/(S-r1))

    main(v0)