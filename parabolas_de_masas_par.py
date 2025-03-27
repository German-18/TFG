import numpy as np
import matplotlib.pyplot as plt

# Datos para Z impares (19, 21, 23, 25)
Z_impar = np.array([19, 21, 23, 25])
E_impar = np.array([43744.849059, 43736.234269, 43738.565331, 43757.941932])

# Datos para Z pares (20, 22, 24)
Z_par = np.array([20, 22, 24])
E_par = np.array([43738.228372, 43735.634985, 43745.990241])

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(Z_impar, E_impar, 'b-o', label='Z impares')
plt.plot(Z_par, E_par, 'r-o', label='Z pares')

plt.xlabel('Z')
plt.ylabel('Energía (MeV/c²)')
plt.title('Energías vs Z (pares e impares)')
plt.grid(True)
plt.legend()
plt.show()