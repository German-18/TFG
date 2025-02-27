import numpy as np
import matplotlib.pyplot as plt

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

# Initialize arrays
a = 0
u = np.array([[Rt, v0]])
x = [a]
iter = 0

# Define the system of differential equations
def fun(x, y):
    return np.array([y[1], -G*Mt/(y[0]**2) + G*Ml/(S-y[0])**2])

d = S - Rl

# RK4 Integration loop
while d - u[iter,0] > 0.1:
    k1 = fun(x[iter], u[iter])
    k2 = fun(x[iter] + h/2, u[iter] + h*k1/2)
    k3 = fun(x[iter] + h/2, u[iter] + h*k2/2)
    k4 = fun(x[iter] + h, u[iter] + h*k3)
    
    # Add new row to u array
    new_u = u[iter] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    u = np.vstack([u, new_u])
    
    print(f"Distance: {u[iter+1,0]}")
    x.append(x[iter] + h)
    iter += 1

tiempo = x[iter]
print(f"Total time: {tiempo} seconds")

# Plotting
plt.figure(figsize=(10,6))
plt.plot(x, u[:,1])
plt.legend(['Velocity'], loc='best')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')
plt.title('Orbital Velocity vs Time')
plt.grid(True)
plt.show()