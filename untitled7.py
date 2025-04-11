import numpy as np
import matplotlib.pyplot as plt

# Параметры для Epsilon Aurigae
G = 4 * np.pi**2  # гравитационная постоянная в а.е.^3 / (солнечная масса * год^2)
m1 = 2.4   # масса Epsilon Aurigae A (в массах Солнца)
m2 = 0.9   # масса Epsilon Aurigae B (в массах Солнца)
M = m1 + m2

# Центр масс
r1_0 = np.array([-m2 / M, 0])  # начальная позиция A
r2_0 = np.array([m1 / M, 0])   # начальная позиция B
v1_0 = np.array([0, -np.sqrt(G * m2 / np.linalg.norm(r1_0 - r2_0))])  # начальная скорость для A
v2_0 = np.array([0, np.sqrt(G * m1 / np.linalg.norm(r1_0 - r2_0))])   # начальная скорость для B

# Параметры интеграции
dt = 0.001  # шаг по времени (лет)
t_max = 27  # орбитальный период (лет)
N = int(t_max / dt)

# Массивы
r1 = np.zeros((N, 2))
r2 = np.zeros((N, 2))
v1 = np.zeros((N, 2))
v2 = np.zeros((N, 2))
t = np.zeros(N)
eclipses = []

# Начальные условия
r1[0] = r1_0
r2[0] = r2_0
v1[0] = v1_0
v2[0] = v2_0

# Интегрирование методом Эйлера
for i in range(N - 1):
    r = r2[i] - r1[i]
    dist = np.linalg.norm(r)
    F = G * m1 * m2 / dist**3 * r  # сила гравитационного притяжения

    # Обновление скоростей и позиций
    v1[i+1] = v1[i] + F / m1 * dt
    v2[i+1] = v2[i] - F / m2 * dt
    r1[i+1] = r1[i] + v1[i+1] * dt
    r2[i+1] = r2[i] + v2[i+1] * dt
    t[i+1] = t[i] + dt

    # Проверка затмения при взгляде вдоль оси Y (проекция на X)
    if abs(r1[i+1][0] - r2[i+1][0]) < 0.01 and abs(r1[i+1][1] - r2[i+1][1]) < 0.05:
        # Считаем затмение, если проекции на X почти совпадают и тела близко по Y
        eclipses.append(t[i+1])

# Визуализация орбит
plt.figure(figsize=(8, 8))
plt.plot(r1[:, 0], r1[:, 1], label="Epsilon Aurigae A", color='b')
plt.plot(r2[:, 0], r2[:, 1], label="Epsilon Aurigae B", color='r')
plt.plot(0, 0, 'ko', label='Центр масс')  # Отметим центр масс
plt.xlabel('X (а.е.)')
plt.ylabel('Y (а.е.)')
plt.legend()
plt.title("Орбиты Epsilon Aurigae A и B")

# Отметим моменты затмений
for e in eclipses:
    idx = int(e / dt)
    plt.plot(r1[idx, 0], r1[idx, 1], 'ro', markersize=4)

plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Выводим первые 10 моментов затмений
print(eclipses[:10])


