import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import seaborn as sns

# 设置字体（如果有中文需求）
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 请根据实际情况修改路径
if os.path.exists(font_path):
    font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# 1. 数据加载和预处理
print("加载数据集...")
train_data = pd.read_csv('trainingData.csv')
test_data = pd.read_csv('validationData.csv')
print("数据集加载完成。")

# 合并数据集
data = pd.concat([train_data, test_data], ignore_index=True)

# 提取Wi-Fi信号强度（前520列）
wifi_signals = data.iloc[:, 0:520]
wifi_signals.columns = [f'AP_{i}' for i in range(1, 521)]  # 重命名列名为 AP 编号

# 将100替换为-110（表示未检测到信号）
wifi_signals.replace(100, -110, inplace=True)

# 添加位置信息
wifi_signals['LONGITUDE'] = data['LONGITUDE']
wifi_signals['LATITUDE'] = data['LATITUDE']
wifi_signals['FLOOR'] = data['FLOOR']
wifi_signals['BUILDINGID'] = data['BUILDINGID']

# 2. 选择信号强度指标

# 方法一：选择特定的 AP
# ap_number = 1  # 选择 AP_1
# wifi_signals['SELECTED_SIGNAL'] = wifi_signals[f'AP_{ap_number}']

# 方法二：计算平均信号强度
wifi_signals['SELECTED_SIGNAL'] = wifi_signals.iloc[:, 0:520].mean(axis=1)

# 方法三：计算最大信号强度
# wifi_signals['SELECTED_SIGNAL'] = wifi_signals.iloc[:, 0:520].max(axis=1)

# 3. 构建三维热力图

# 将楼层映射为高度（假设每层楼高为3米）
floor_height = {0: 0, 1: 3, 2: 6, 3: 9, 4: 12}
wifi_signals['HEIGHT'] = wifi_signals['FLOOR'].map(floor_height)

# 为了减少数据量，随机采样部分数据
sample_size = 5000  # 您可以根据需要调整样本数量
if len(wifi_signals) > sample_size:
    wifi_signals = wifi_signals.sample(n=sample_size, random_state=42).reset_index(drop=True)

# 获取坐标和信号强度
x = wifi_signals['LONGITUDE']
y = wifi_signals['LATITUDE']
z = wifi_signals['HEIGHT']
signal = wifi_signals['SELECTED_SIGNAL']

# 4. 可视化

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 规范化信号强度用于颜色映射
norm = Normalize(vmin=signal.min(), vmax=signal.max())
colors = plt.cm.jet(norm(signal))  # 使用 jet 颜色映射，从蓝（低）到红（高）

# 绘制散点图
scatter = ax.scatter(x, y, z, c=colors, marker='o', s=15, edgecolor='k')

# 设置坐标轴标签和标题
ax.set_xlabel('经度', fontproperties=font_prop)
ax.set_ylabel('纬度', fontproperties=font_prop)
ax.set_zlabel('高度（米）', fontproperties=font_prop)
ax.set_title('三维热力图：Wi-Fi 信号强度分布', fontproperties=font_prop)

# 添加颜色条
mappable = plt.cm.ScalarMappable(cmap='jet', norm=norm)
mappable.set_array([])
cbar = plt.colorbar(mappable, shrink=0.5, aspect=10)
cbar.set_label('信号强度', fontproperties=font_prop)

plt.show()
