import numpy as np
import matplotlib.pyplot as plt

# === Параметры модели ===
depth_primary = 0.15     # глубина первичного затмения
depth_secondary = 0.03   # глубина вторичного затмения
width = 0.05             # ширина затмений (в фазе)
phase_shift = 0.5        # вторичное затмение на половине орбиты

# === Фазовая шкала (0 ... 1) ===
phase = np.linspace(0, 1, 1000)
flux = np.ones_like(phase)

# === Добавим первичное затмение в фазе ~0
primary = np.abs(phase - 0.0) < width / 2
flux[primary] -= depth_primary

# === Добавим вторичное затмение в фазе ~0.5
secondary = np.abs(phase - phase_shift) < width / 2
flux[secondary] -= depth_secondary

# === Симулируем реальные данные с шумом
np.random.seed(42)
obs_phase = np.random.uniform(0, 1, 300)
obs_flux = 1.0 - 0.005 * np.random.randn(300)

# Добавим немного реального затмения в наблюдение (для примера)
obs_flux[(np.abs(obs_phase - 0.0) < width / 2)] -= depth_primary
obs_flux[(np.abs(obs_phase - phase_shift) < width / 2)] -= depth_secondary

# === Построим график ===
plt.figure(figsize=(10, 5))
plt.scatter(obs_phase, obs_flux, color='cornflowerblue', s=10, alpha=0.7, label='Наблюд. данные (фаза)')
plt.plot(phase, flux, color='darkorange', lw=2, label='Синтетическая модель')

plt.xlabel("Фаза")
plt.ylabel("Яркость (отн. ед.)")
plt.title("Фазовая кривая: синтетическая модель + наблюдение (Epsilon Aurigae)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


