import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import TextBox, Button
import threading
import time

# 指定字体，解决中文乱码问题
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

# 生成数据
np.random.seed(0)
x = np.linspace(0, 10, 50)
y = 2 * x + 1 + np.random.normal(scale=2, size=x.shape)

# 定义斜率 m 的范围
m_values = np.linspace(0, 4, 1000)
cost_values = []
for m in m_values:
    y_pred = m * x
    cost = np.mean((y - y_pred) ** 2) / 2  # 均方误差的一半
    cost_values.append(cost)
cost_values = np.array(cost_values)

# 创建一个包含3个子区域的图形窗口
fig = plt.figure(figsize=(14, 6))  # 调整图形窗口大小
gs_main = fig.add_gridspec(1, 3, width_ratios=[0.3, 1, 1], wspace=0.3)  # 控件区域宽度减小

# 左侧控件区域
ax_controls = fig.add_subplot(gs_main[0, 0])
ax_controls.axis('off')  # 隐藏坐标轴

# 中间区域用于绘制成本函数
ax1 = fig.add_subplot(gs_main[0, 1])
ax1.plot(m_values, cost_values, label='成本函数 J(m)')
ax1.set_xlabel('斜率 m', fontproperties=font)
ax1.set_ylabel('成本 J(m)', fontproperties=font)
ax1.set_title('成本函数', fontproperties=font)
ax1.legend(prop=font)

# 右侧区域用于绘制线性拟合
ax2 = fig.add_subplot(gs_main[0, 2])
ax2.scatter(x, y, label='数据点')
line_fit, = ax2.plot(x, y, 'r-', label='拟合直线')
ax2.set_xlabel('x', fontproperties=font)
ax2.set_ylabel('y', fontproperties=font)
ax2.set_title('线性拟合', fontproperties=font)
ax2.legend(prop=font)

# 初始化斜率 m 值
m_init = 2
m_current = m_init
y_pred_init = m_current * x
line_fit.set_ydata(y_pred_init)
point_on_curve, = ax1.plot([m_current], [np.mean((y - m_current * x) ** 2) / 2], 'ro', markersize=8)

# 定义一个可拖动点的类
class DraggablePoint:
    def __init__(self, ax, point):
        self.ax = ax
        self.point = point
        self.press = None
        self.connect()

    def connect(self):
        '连接所有需要的事件'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if event.button != 1: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.get_data(), event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return
        (x0, y0), xpress, ypress = self.press
        dx = event.xdata - xpress
        m_current = x0[0] + dx
        update_fit(m_current)  # 更新拟合线和成本点
        self.point.set_data([m_current], [np.mean((y - m_current * x) ** 2) / 2])
        self.point.figure.canvas.draw()

    def on_release(self, event):
        '释放鼠标时重置背景'
        self.press = None
        self.point.figure.canvas.draw()

    def disconnect(self):
        '断开所有连接'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

# 更新拟合直线和成本点的函数
def update_fit(m_current):
    y_pred_new = m_current * x
    line_fit.set_ydata(y_pred_new)
    line_fit.set_label(f'y = {m_current:.2f} x')
    ax2.legend(prop=font)
    fig.canvas.draw_idle()

# 使点可拖动
draggable_point = DraggablePoint(ax1, point_on_curve)

# 开始绘图显示
plt.show()
