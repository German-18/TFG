import numpy as np
import matplotlib.pyplot as plt

# Datos
Z = np.array([19, 20, 21, 22, 23, 24, 25])
E = np.array([43744.849059, 43738.228372, 43736.234269, 43735.634985, 
              43738.565331, 43745.990241, 43757.941932])

# Ajuste polinómico de grado 2 (parábola)
coef = np.polyfit(Z, E, 2)
Z_fit = np.linspace(19, 25, 100)
E_fit = np.polyval(coef, Z_fit)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(Z_fit, E_fit, 'b-', label='Ajuste parabólico')
plt.plot(Z, E, 'ro', label='Datos experimentales')

plt.xlabel('Z')
plt.ylabel('Energía (MeV/c²)')
plt.title('Parábola de masas')
plt.grid(True)
plt.legend()
plt.show()
