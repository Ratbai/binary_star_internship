import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Параметры для Эпсилон Аргу ===
G = 39.478  # Гравитационная постоянная в а.е.^3 / (M☉ * год^2)
m1 = 1.08  # Масса Эпсилон Аргу A
m2 = 0.94  # Масса Эпсилон Аргу B
a = 0.115  # Большая полуось орбиты в а.е.
e = 0.397  # Эксцентриситет орбиты
T = 101.2 / 365.25  # Период орбиты в годах

# Начальные условия для расчета орбит
r0 = a * (1 - e)
x1_0 = -r0 * m2 / (m1 + m2)
x2_0 = r0 * m1 / (m1 + m2)
v0 = np.sqrt(G * (m1 + m2) * (1 + e) / (a * (1 - e)))
vx1_0, vx2_0 = 0, 0
vy1_0 = v0 * m2 / (m1 + m2)
vy2_0 = -v0 * m1 / (m1 + m2)
y0 = [x1_0, 0, 0, x2_0, 0, 0, vx1_0, vy1_0, 0, vx2_0, vy2_0, 0]

# === Уравнения движения ===
def equations(t, state):
    x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = state
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    ax1 = G * m2 * dx / r**3
    ay1 = G * m2 * dy / r**3
    az1 = G * m2 * dz / r**3
    ax2 = -G * m1 * dx / r**3
    ay2 = -G * m1 * dy / r**3
    az2 = -G * m1 * dz / r**3
    return [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]

# === Интегрирование ===
t_eval = np.linspace(0, T, 5000)  # Вектор времени
sol = solve_ivp(equations, (0, T), y0, t_eval=t_eval)
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]

# === Физические параметры ===
R_sun_au = 0.00465047  # Радиус Солнца в а.е.
R1 = 1.05 * R_sun_au  # Радиус Эпсилон Аргу A
R2 = 0.95 * R_sun_au  # Радиус Эпсилон Аргу B
L1 = 1.16  # Светимость Эпсилон Аргу A
L2 = 0.80  # Светимость Эпсилон Аргу B

# === Функция перекрытия двух кругов ===
def overlap_area(Ra, Rb, d):
    if d >= Ra + Rb:
        return 0  # Нет перекрытия
    elif d <= abs(Ra - Rb):
        return np.pi * min(Ra, Rb)**2  # Полное затмение
    else:
        part1 = Ra**2 * np.arccos((d**2 + Ra**2 - Rb**2) / (2 * d * Ra))
        part2 = Rb**2 * np.arccos((d**2 + Rb**2 - Ra**2) / (2 * d * Rb))
        part3 = 0.5 * np.sqrt((-d + Ra + Rb) * (d + Ra - Rb) * (d - Ra + Rb) * (d + Ra + Rb))
        return part1 + part2 - part3

# === Вычисление яркости с учетом перекрытия ===
brightness = []
full_brightness = L1 + L2  # Полная яркость, когда оба видны

for i in range(len(t_eval)):
    dx = x2[i] - x1[i]
    dz = z2[i] - z1[i]
    d_proj = np.sqrt(dx**2 + dz**2)  # Расстояние по проекции XZ
    # Вычисляем площадь перекрытия
    A_overlap = overlap_area(R1, R2, d_proj)
    A1 = np.pi * R1**2  # Площадь Эпсилон Аргу A
    A2 = np.pi * R2**2  # Площадь Эпсилон Аргу B
    # Проверка, кто ближе (по оси Y)
    if y1[i] > y2[i]:  # Эпсилон Аргу B ближе, может закрыть Эпсилон Аргу A
        blocked_fraction = A_overlap / A1
        total_L = L1 * (1 - blocked_fraction) + L2
    elif y2[i] > y1[i]:  # Эпсилон Аргу A ближе, может закрыть Эпсилон Аргу B
        blocked_fraction = A_overlap / A2
        total_L = L1 + L2 * (1 - blocked_fraction)
    else:
        total_L = full_brightness
    brightness.append(total_L)

brightness = np.array(brightness)

# === Построение синтетической кривой блеска ===
plt.figure(figsize=(10, 5))
plt.plot(t_eval * 365.25, brightness, color='darkorange')  # Переводим время в дни
plt.axhline(full_brightness, color='gray', linestyle='--', label='Полная яркость (без затмений)')
plt.xlabel('Время (дни)')
plt.ylabel('Яркость (отн. ед.)')
plt.title('Синтетическая кривая блеска Эпсилон Аргу')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
