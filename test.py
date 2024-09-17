import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 2D data: two concentric circles
np.random.seed(0)
num_samples = 500

# Inner circle (Class 1)
theta_inner = 2 * np.pi * np.random.rand(num_samples // 2)
r_inner = 5 + np.random.randn(num_samples // 2)
x_inner = r_inner * np.cos(theta_inner)
y_inner = r_inner * np.sin(theta_inner)

# Outer circle (Class 2)
theta_outer = 2 * np.pi * np.random.rand(num_samples // 2)
r_outer = 10 + np.random.randn(num_samples // 2)
x_outer = r_outer * np.cos(theta_outer)
y_outer = r_outer * np.sin(theta_outer)

# Combine data
X = np.concatenate((np.vstack((x_inner, y_inner)).T,
                    np.vstack((x_outer, y_outer)).T))
Y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Plot 2D data
plt.figure(figsize=(6, 6))
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', label='Class 1')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class 2')
plt.legend()
plt.title('Non-linearly Separable Data in 2D Space')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.close()

# Define mapping function to 3D space: z = x^2 + y^2
def mapping(X):
    z = X[:, 0]**2 + X[:, 1]**2
    return np.hstack((X, z.reshape(-1, 1)))

# Map to 3D space
X_mapped = mapping(X)

# Compute z-values for each class
z_class1 = X_mapped[Y == 0][:, 2]
z_class2 = X_mapped[Y == 1][:, 2]

# Compute max and min z-values
z_class1_max = z_class1.max()
z_class2_min = z_class2.min()

# Calculate the threshold z-value
z_threshold = (z_class1_max + z_class2_min) / 2

# Optional: Print the computed values
print(f"Class 1 max z-value: {z_class1_max}")
print(f"Class 2 min z-value: {z_class2_min}")
print(f"Separating plane at z = {z_threshold}")

# Plot 3D data
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_mapped[Y == 0][:, 0], X_mapped[Y == 0][:, 1], X_mapped[Y == 0][:, 2],
           color='red', label='Class 1')
ax.scatter(X_mapped[Y == 1][:, 0], X_mapped[Y == 1][:, 1], X_mapped[Y == 1][:, 2],
           color='blue', label='Class 2')

# Plot separating plane at z = z_threshold
xx, yy = np.meshgrid(np.linspace(-15, 15, 10), np.linspace(-15, 15, 10))
zz = np.full_like(xx, z_threshold)
ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('Linearly Separable Data in 3D Space')
plt.legend()
plt.show()
plt.close()
