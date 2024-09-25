import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成更加复杂的非线性数据
X = np.linspace(-3, 3, 100)
Y = 3 * X**2 + 2 * X + np.random.randn(*X.shape) * 5  # 非线性函数，增加噪声

# w 和 b 的取值范围
w = np.linspace(0, 6, 100)
b = np.linspace(-5, 5, 100)

# 生成网格
W, B = np.meshgrid(w, b)

# 计算损失函数，并引入正则化项
def compute_cost(W, B, X, Y, lambda_reg=0.1):
    Y_pred = W * X[:, np.newaxis, np.newaxis] + B
    cost = np.mean((Y_pred - Y)**2, axis=0)
    # 加入正则化项
    reg_term = lambda_reg * (W**2 + B**2)
    return cost + reg_term

Cost = compute_cost(W, B, X, Y)

# 创建图形并设置大小
fig = plt.figure(figsize=(14, 6))

# 添加散点图
ax1 = fig.add_subplot(121)
ax1.scatter(X, Y, color='blue')
ax1.set_title('Data Scatter Plot (Non-linear)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# 添加 3D 曲面图
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(W, B, Cost, cmap='coolwarm', edgecolor='none')
ax2.set_title('Enhanced Cost Function Surface')
ax2.set_xlabel('Weight w')
ax2.set_ylabel('Bias b')
ax2.set_zlabel('Cost')

# 添加颜色条
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
