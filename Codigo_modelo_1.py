import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
h = 0.5
G = 6.6726e-11
Mt = 5.975e24
Ml = 7.35e22
Rt = 6.378e6
Rl = 1.738e6
S = 3.844e8

# Initial calculations
r1 = S * (np.sqrt(Mt)/(np.sqrt(Mt) + np.sqrt(Ml)))
v0 = np.sqrt(2) * np.sqrt(G*Mt/Rt - G*Ml/(S-Rt) - G*Mt/r1 + G*Ml/(S-r1))

# Define the system of differential equations
def system(t, y):
    r, v = y
    return [v, -G*Mt/(r**2) + G*Ml/(S-r)**2]

# Initial conditions
y0 = [Rt, v0]  # [r(0), v(0)]
t_span = (0, 282440)  # Intervalo de tiempo
t_eval = np.linspace(t_span[0], t_span[1], 282440)

# Solve IVP
sol = solve_ivp(system,t_span,y0,method='RK45',t_eval=t_eval,rtol=1e-8,atol=1e-8)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[1])  # sol.y[0] es r(t), sol.y[1] es v(t)
plt.legend(['Velocity'], loc='best')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')
plt.title('Orbital Velocity vs Time')
plt.grid(True)
plt.show()