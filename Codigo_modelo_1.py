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

    # Initial conditions
    y0 = [Rt, v0]  # [r(0), v(0)]
    t_span = (0, 400000)  # Intervalo de tiempo

    # Solve IVP
    sol = solve_ivp(derivadas,t_span,y0,method='RK45',rtol=1e-8,atol=1e-8,events=collision)

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

# Define the system of differential equations
def derivadas(t, y):
    r, v = y
    return [v, aceleracion(t, r)]

def aceleracion(t, r):
    return -G*Mt/(r**2) + G*Ml/(S-r)**2

# Definición de función eventos que detecta colision con la Luna
def collision(t, y):
    disSupMoon = S-y[0]-Rl #Distancia entre el satélite y la Luna
    disSupEarth = y[0]-Rt #Distancia entre el satélite y la Tierra
    return [disSupMoon,disSupEarth]
collision.terminal = True # La integración se detiene cuando la función de eventos es cero

# Call main function
if __name__ == '__main__':
    
    # calculation of initial velocity
    r1 = S * (np.sqrt(Mt)/(np.sqrt(Mt) + np.sqrt(Ml)))
    v0 = np.sqrt(2) * np.sqrt(G*Mt/Rt - G*Ml/(S-Rt) - G*Mt/r1 + G*Ml/(S-r1))

    main(v0)