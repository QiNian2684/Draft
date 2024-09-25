import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 设置字体（如果有中文需求）
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 请根据实际情况修改路径
if os.path.exists(font_path):
    font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# 1. 数据加载
train_data = pd.read_csv('trainingData.csv')
test_data = pd.read_csv('validationData.csv')

# 2. 数据预处理和模型训练
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 提取特征和标签
X_train = train_data.iloc[:, 0:520]
X_test = test_data.iloc[:, 0:520]
y_train = train_data['FLOOR']
y_test = test_data['FLOOR']

# 替换缺失值
X_train.replace(100, -110, inplace=True)
X_test.replace(100, -110, inplace=True)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练
svm_clf = SVC(kernel='linear', C=1)
svm_clf.fit(X_train, y_train)

# 模型预测
y_pred = svm_clf.predict(X_test)

# 3. 数据合并
test_positions = test_data[['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']].reset_index(drop=True)
test_positions['PREDICTION'] = y_pred

# 将楼层映射为高度（假设每层楼高为3米）
floor_height = {0: 0, 1: 3, 2: 6, 3: 9, 4: 12}
test_positions['ACTUAL_HEIGHT'] = test_positions['FLOOR'].map(floor_height)
test_positions['PREDICTED_HEIGHT'] = test_positions['PREDICTION'].map(floor_height)

# 获取建筑物列表
buildings = test_positions['BUILDINGID'].unique()

# 4. 分楼栋展示 3D 图，并在真实值与预测值之间连虚线
for building in buildings:
    building_data = test_positions[test_positions['BUILDINGID'] == building].reset_index(drop=True)

    # 如果数据过多，随机采样一部分
    sample_size = 200  # 您可以根据需要调整样本数量
    if len(building_data) > sample_size:
        sample_indices = np.random.choice(building_data.index, size=sample_size, replace=False)
        building_data = building_data.loc[sample_indices].reset_index(drop=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制实际位置
    ax.scatter(building_data['LONGITUDE'], building_data['LATITUDE'], building_data['ACTUAL_HEIGHT'],
               c='b', marker='o', s=50, label='实际值')

    # 绘制预测位置
    ax.scatter(building_data['LONGITUDE'], building_data['LATITUDE'], building_data['PREDICTED_HEIGHT'],
               c='r', marker='^', s=50, label='预测值')

    # 在实际值与预测值之间绘制虚线
    for i in range(len(building_data)):
        x = building_data.loc[i, 'LONGITUDE']
        y = building_data.loc[i, 'LATITUDE']
        z_actual = building_data.loc[i, 'ACTUAL_HEIGHT']
        z_pred = building_data.loc[i, 'PREDICTED_HEIGHT']
        ax.plot([x, x], [y, y], [z_actual, z_pred], c='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('经度', fontproperties=font_prop)
    ax.set_ylabel('纬度', fontproperties=font_prop)
    ax.set_zlabel('高度（米）', fontproperties=font_prop)
    ax.set_title(f'建筑物 {building} 的实际值与预测值比较', fontproperties=font_prop)
    ax.legend(prop=font_prop)
    plt.show()
