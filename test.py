
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Параметры модели
alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

def lotka_volterra(t, z):
    x, y = z  # распаковываем вектор состояния
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Начальные условия и интервал времени
z0 = [10.0, 5.0]
t_span = (0, 20)
t_eval = np.linspace(0, 20, 400)  # точки, в которых хотим получить решение

# Решение системы
sol = solve_ivp(lotka_volterra, t_span, z0, t_eval=t_eval, method='RK45')

# Проверка на успешность решения
if not sol.success:
    raise RuntimeError(sol.message)

# Извлекаем решения для x и y
x = sol.y[0]   # жертвы
y = sol.y[1]   # хищники
t = sol.t

# График: численность во времени
plt.figure(figsize=(8, 5))
plt.plot(t, x, 'b-', label='Жертвы (x)')
plt.plot(t, y, 'r-', label='Хищники (y)')
plt.xlabel('Время t')
plt.ylabel('Численность')
plt.title('Модель Лотки–Вольтерра: динамика популяций')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Дополнительно: фазовый портрет (x vs y)
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'k-', linewidth=1)
plt.xlabel('Жертвы (x)')
plt.ylabel('Хищники (y)')
plt.title('Фазовый портрет системы')
plt.grid(True)
plt.tight_layout()
plt.show()

