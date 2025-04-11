# two_body_model.py
# Этап 1: решение задачи двух тел для системы Epsilon Aurigae

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === 1. Параметры для Epsilon Aurigae ===
G = 39.478  # Гравитационная постоянная в а.е.^3 / (M☉ * год^2)
m1 = 2.4    # Масса звезды A (F-супергигант), M☉
m2 = 0.9    # Масса звезды B (диск + возможная звезда), M☉
a = 5.0     # Большая полуось орбиты в а.е.
e = 0.2     # Эксцентриситет
T = 27.0    # Орбитальный период (в годах)

# === 2. Начальные условия (перицентр) ===
r0 = a * (1 - e)
x1_0, y1_0, z1_0 = -r0 * m2 / (m1 + m2), 0, 0
x2_0, y2_0, z2_0 = r0 * m1 / (m1 + m2), 0, 0
v0 = np.sqrt(G * (m1 + m2) * (1 + e) / (a * (1 - e)))
vx1_0, vy1_0, vz1_0 = 0, v0 * m2 / (m1 + m2), 0
vx2_0, vy2_0, vz2_0 = 0, -v0 * m1 / (m1 + m2), 0

# === 3. Функция системы уравнений ===
def equations(t, state):
    x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = state
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    ax1 = G * m2 * (x2 - x1) / r12**3
    ay1 = G * m2 * (y2 - y1) / r12**3
    az1 = G * m2 * (z2 - z1) / r12**3
    ax2 = -G * m1 * (x2 - x1) / r12**3
    ay2 = -G * m1 * (y2 - y1) / r12**3
    az2 = -G * m1 * (z2 - z1) / r12**3
    return [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]

# === 4. Интегрирование ===
t_span = (0, T)
y0 = [x1_0, y1_0, z1_0, x2_0, y2_0, z2_0, vx1_0, vy1_0, vz1_0, vx2_0, vy2_0, vz2_0]
t_eval = np.linspace(0, T, 5000)
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# === 5. Получение координат ===
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]

# === 6. Построение орбит ===
plt.figure(figsize=(12, 5))

# Вид сверху (XY)
plt.subplot(1, 2, 1)
plt.plot(x1, y1, label='Epsilon Aurigae A', color='blue')
plt.plot(x2, y2, label='Epsilon Aurigae B', color='red')
plt.scatter(0, 0, color='black', marker='x', label='Центр масс')
plt.xlabel('X (а.е.)')
plt.ylabel('Y (а.е.)')
plt.title('Орбиты — вид сверху (XY)')
plt.legend()

# Вид сбоку (XZ)
plt.subplot(1, 2, 2)
plt.plot(x1, z1, label='Epsilon Aurigae A', color='blue')
plt.plot(x2, z2, label='Epsilon Aurigae B', color='red')
plt.scatter(0, 0, color='black', marker='x', label='Центр масс')
plt.xlabel('X (а.е.)')
plt.ylabel('Z (а.е.)')
plt.title('Орбиты — вид сбоку (XZ)')
plt.legend()

plt.tight_layout()
plt.savefig("figures/orbits_epsilon_aurigae.png")  # сохраняем график
plt.show()
