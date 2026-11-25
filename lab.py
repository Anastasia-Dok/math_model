import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# 1. МОДЕЛЬ
# =============================================================================
def agrarian_model(t, state, p):
    X, Y, N = state
    # Простая защита
    X = max(X, 0.0); Y = max(Y, 1e-3); N = max(N, 1e-3)
    
    # Распаковка параметров
    alpha, Ax, Ay = p['alpha'], p['Ax'], p['Ay']
    C1, C2 = p['C1'], p['C2']
    f, b, X0 = p['f'], p['b'], p['X0']
    R0, N_star = p['R0'], p['N_star']
    r, Y0 = p['r'], p['Y0']

    # Упрощенные функции (убрали лишние условия для скорости)
    R_N = R0 if N <= N_star else R0 * (N_star / N)
    f_X = f * (1.0 + b * (X**2 / (X0**2 + X**2 + 1e-6)))
    F = f_X * R_N * N
    
    G = alpha * F 
    Qx = Ax * X
    Qy = Ay * Y
    C = (C1 / Y + C2 * Y) * N
    D = r * (1.0 - Y0 / Y)

    dXdt = G - Qx - C if X > 0 else G
    dNdt = N * D
    dYdt = (F / N) - Qy - (G / N) - Y * D
    
    return [dXdt, dYdt, dNdt]

# Параметры
base_params = {
    'alpha': 0.22, 'Ax': 0.04, 'Ay': 0.25,
    'C1': 0.03, 'C2': 0.005, 'f': 1.8, 'b': 0.6,
    'X0': 300.0, 'R0': 2.0, 'N_star': 400.0,
    'r': 0.06, 'Y0': 1.0
}

state_init = [100.0, 1.3, 120.0]

# Функция для быстрого решения (уменьшены точки, добавлен max_step)
def fast_solve(t_span, y0, p):
    return solve_ivp(agrarian_model, t_span, y0, args=(p,), 
                     t_eval=np.linspace(t_span[0], t_span[1], 500), 
                     method='RK45', max_step=1.0, rtol=1e-4)

# =============================================================================
# 2. СЦЕНАРИИ (Склейка)
# =============================================================================
print("Считаем сценарий 1...")
# --- Сценарий 1: Засуха ---
s1_b = fast_solve((0, 150), state_init, base_params)
p_dry = base_params.copy(); p_dry['f'] = 0.4
s1_c = fast_solve((150, 200), s1_b.y[:, -1], p_dry)
p_rec = base_params.copy(); p_rec['f'] = 2.0
s1_a = fast_solve((200, 400), s1_c.y[:, -1], p_rec)
# Склейка
t1 = np.concatenate((s1_b.t, s1_c.t, s1_a.t))
y1 = np.concatenate((s1_b.y, s1_c.y, s1_a.y), axis=1)

print("Считаем сценарий 2...")
# --- Сценарий 2: Чума ---
s2_b = fast_solve((0, 150), state_init, base_params)
X_p, Y_p, N_p = s2_b.y[:, -1]
N_p *= 0.45; Y_p *= 1.5  # Удар и наследование
s2_a = fast_solve((150, 400), [X_p, Y_p, N_p], base_params)
t2 = np.concatenate((s2_b.t, s2_a.t))
y2 = np.concatenate((s2_b.y, s2_a.y), axis=1)


# =============================================================================
# 3. ОТРИСОВКА (Упрощенная, 1 окно с 3 графиками для скорости)
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# График 1: Засуха
axs[0].plot(y1[2], y1[0], 'k-', lw=1)
axs[0].axvspan(s1_b.y[2,-1], s1_c.y[2,-1], color='red', alpha=0.2)
axs[0].set_title('1. Засуха (Низкие Нилы)')
axs[0].set_xlabel('Население N'); axs[0].set_ylabel('Казна X'); axs[0].grid(True)

# График 2: Чума
axs[1].plot(y2[2], y2[0], 'k-', lw=1)
axs[1].scatter(N_p, X_p, color='red', zorder=5) # Точка удара
axs[1].set_title('2. Чума')
axs[1].set_xlabel('Население N'); axs[1].grid(True)



plt.tight_layout()
plt.show()

# params = {
#     'alpha': 0.15,     # Доля изъятия продукта государством (налог)
#     'Ax': 0.04,        # Коэффициент трат элиты
#     'Ay': 0.35,        # Базовое потребление крестьян
#     'C1': 0.05,        # Базовые издержки управления
#     'C2': 0.005,       # Дополнительные издержки от зажиточности
#     'f': 1.0,          # Базовая производительность труда
#     'b': 0.6,          # Сила влияния госинфраструктуры на урожай
#     'X0': 300.0,       # Масштаб госсредств для полунасыщения инвестиций (важно!)
#     'R0': 2.0,         
#     'N_star': 400.0,   # Оптимальное население в масштабе сотен
#     'r': 0.25,         # Скорость демографического роста
#     'Y0': 1.0          # Критический прожиточный минимум           
# }

# # --- 2. Функция системы дифференциальных уравнений ---
# def agrarian_model(t, state, p):
#     X, Y, N = state
    
#     # Защита от деления на ноль
#     N = max(N, 1e-5)
    
    
#     # Распаковка параметров
#     alpha, Ax, Ay = p['alpha'], p['Ax'], p['Ay']
#     C1, C2 = p['C1'], p['C2']
#     f, b, X0 = p['f'], p['b'], p['X0']
#     R0, N_star = p['R0'], p['N_star']
#     r, Y0 = p['r'], p['Y0']

#     # Вычисление функций 
    
#     #alpha * X * Y * N #налоги
#     Qx = Ax * X #потреб гос-ва 
#     Qy = Ay * Y #потреб крестьян
#     C = (C1 / max(Y, 1e-5) + C2 * Y) * N #затраты на управление
    
#     # Функция производительности f(X) 
#     f_X = f * (1 + b * (X**2 / (X0**2 + X**2))) #вклад гос-ва в производительность труда
    
#     # Функция площади R(N)
#     R_N = R0 if N <= N_star else R0 * (N_star / N) 
    
#     F = f_X * R_N * N #характ совокупное с/х производство гос-ва
#     G =alpha * F 
#     # Демографическая функция D(Y)
#     D = r * (1 - Y0 / max(Y, 1e-5)) if Y > 0 else 0

#     # Производные
#     dXdt = G - Qx - C
#     dNdt = N * D
#     dYdt = (F / N) - Qy - (G / N) - Y * D
    
#     return [dXdt, dYdt, dNdt]
    
   

# # --- 3. Начальные условия и время ---
# X_init = 420.0   # Средний уровень госказны
# Y_init = 2.0    # Накопления крестьян чуть выше критического Y0=1.0
# N_init = 1900.0  # Население чуть выше оптимума N_star=100 (запускаем мальтузианское давление)

# state0 = [X_init, Y_init, N_init]

# t_span = (0, 200)
# t_eval = np.linspace(t_span[0], t_span[1], 15000)

# # --- 4. Решение системы ---
# sol = solve_ivp(
#     fun=lambda t, y: agrarian_model(t, y, params),
#     t_span=t_span,
#     y0=state0,
#     t_eval=t_eval,
#     method='RK45',
#     max_step=0.02
# )

# # --- 5. Обработка результатов ---
# X_sol = sol.y[0]
# Y_sol = sol.y[1]
# N_sol = sol.y[2]


# # --- 6. Визуализация (аналогично Рис. 2.9) ---
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# # График X(t)
# axs[0, 0].plot(sol.t, X_sol, color='blue')
# axs[0, 0].set_title('Изменение средств государства X(t)')
# axs[0, 0].set_xlabel('Время')
# axs[0, 0].set_ylabel('X')
# axs[0, 0].grid(True)

# # График Y(t)
# axs[0, 1].plot(sol.t, Y_sol, color='green')
# axs[0, 1].set_title('Средние накопления крестьян Y(t)')
# axs[0, 1].set_xlabel('Время')
# axs[0, 1].set_ylabel('Y')
# axs[0, 1].grid(True)

# # График N(t)
# axs[1, 0].plot(sol.t, N_sol, color='red')
# axs[1, 0].set_title('Численность крестьян N(t)')
# axs[1, 0].set_xlabel('Время')
# axs[1, 0].set_ylabel('N')
# axs[1, 0].grid(True)

# # Фазовый портрет (X, N)
# axs[1, 1].plot(N_sol, X_sol, color='purple')
# axs[1, 1].set_title('Фазовый портрет (X, N)')
# axs[1, 1].set_ylabel('X (средства государства)')
# axs[1, 1].set_xlabel('N (население)')
# axs[1, 1].grid(True)

# plt.tight_layout()
# plt.show()