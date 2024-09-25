import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 设置随机种子，保证结果可重复
np.random.seed(0)

# 样本数量
num_samples = 100

# 生成类别0的数据（均值为(2,2)，协方差为[[1, 0.75], [0.75, 1]]）
mean0 = [2, 2]
cov0 = [[1, 0.75], [0.75, 1]]
X0 = np.random.multivariate_normal(mean0, cov0, num_samples)
y0 = np.zeros(num_samples)

# 生成类别1的数据（均值为(4,4)，协方差为[[1, 0.75], [0.75, 1]]）
mean1 = [4, 4]
cov1 = [[1, 0.75], [0.75, 1]]
X1 = np.random.multivariate_normal(mean1, cov1, num_samples)
y1 = np.ones(num_samples)

# 合并数据
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# 为特征添加偏置项（截距）
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    # 为了避免log(0)，添加一个小的常数epsilon
    epsilon = 1e-5
    cost = (-1 / m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost

plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='b', label='类别 0')
plt.scatter(X1[:, 0], X1[:, 1], c='r', label='类别 1')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('样本数据散点图')
plt.legend()
plt.show()

# 定义theta1和theta2的取值范围
theta1_vals = np.linspace(-5, 5, 50)
theta2_vals = np.linspace(-5, 5, 50)

# 初始化代价函数值矩阵
J_vals = np.zeros((len(theta1_vals), len(theta2_vals)))

# 固定theta0为0
theta0 = 0

# 计算每个(theta1, theta2)组合下的代价函数值
for i, theta1 in enumerate(theta1_vals):
    for j, theta2 in enumerate(theta2_vals):
        theta = np.array([theta0, theta1, theta2])
        J_vals[i, j] = compute_cost(theta, X_with_intercept, y)

# 创建网格，方便绘图
Theta1, Theta2 = np.meshgrid(theta1_vals, theta2_vals)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 由于meshgrid的索引方式，需要转置J_vals
ax.plot_surface(Theta1, Theta2, J_vals.T, cmap='viridis')
ax.set_xlabel('Theta1')
ax.set_ylabel('Theta2')
ax.set_zlabel('代价函数值')
ax.set_title('代价函数三维可视化')
plt.show()
