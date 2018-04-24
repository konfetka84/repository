import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

# 0. Перейти в нужный каталог, проверить, что он текущий
os.chdir('D:/lab1/py')
os.getcwd()

# 1. Создать скаляр, массив, матрицу
a = 10
a_rand = np.random.randint(10)
a_zero = 0
a_ones = 1

b = np.array([1, 2, 3])
b_rand = np.random.randint(10, size=5)
b_zero = np.zeros(5)
b_ones = np.ones(5)

c = np.array([[1, 2, 3], [4, 5, 6]])
c_rand = np.random.randint(10, size=(2, 3))
c_zero = np.zeros((2, 3))
c_ones = np.ones((2, 3))

# 2. Загрузить данные из текстового файла
data = np.loadtxt('data.txt')
print(data)
# 3. Загрузить одномерные данные и оценить параметры
data = scipy.io.loadmat('data/1D/var3.mat')
print(data.keys())
n = data['n']
# можно вывести data и посмотреть, какие переменные там сохранены
# например, в файле var3.mat в data['n'] хранится вектор-столбец из 1000 элементов
max_    = np.max(n)     # максимальное
print(max_)
min_    = np.min(n)     # минимальное
print(min_)
mean_   = np.mean(n)    # среднее
print(mean_)
median_ = np.median(n)  # медиана
print(median_)
var_    = np.var(n)     # дисперсия
print(var_)
std_    = np.std(n)     # среднеквадратическое отклонение
print(std_)

# 4. Вывести графики одномерной случайной величины и плотность распределения
plt.plot(n)
plt.grid() 
plt.hlines(mean_, 0, len(n), colors='r', linestyles='solid')
plt.hlines(mean_ + std_, 0, len(n), colors='g', linestyles='dashed')
plt.hlines(mean_ - std_, 0, len(n), colors='g', linestyles='dashed')
plt.show()

plt.hist(n, bins=10)
plt.grid() 
plt.show()

# 5. Автокорреляция случайной величины на графике
def autocorrelate(a):
  n = len(a)
  cor = []
  for i in range(n//2, n//2+n):
    a1 = a[:i+1]   if i < n else a[i-n+1:]
    a2 = a[n-i-1:] if i < n else a[:2*n-i-1]
    cor.append(np.corrcoef(a1, a2)[0, 1])
  return np.array(cor)

n_1d = np.ravel(n)
cor = autocorrelate(n_1d)
plt.plot(cor)
plt.show()

# 6. Загрузить многомерные данные
data = scipy.io.loadmat('data/ND/var3.mat')
print(data.keys())
mn = data['mn']

# 7. Построить матрицу корреляции
n = mn.shape[1]
corr_matrix = np.zeros((n, n))

for i in range(0, n):
  for j in range(0, n):
    corr_matrix[i, j] = np.corrcoef(mn[:, i], mn[:, j])[0, 1]

np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(mn[:, 2], mn[:, 5], 'b.')
plt.grid()
plt.show()
