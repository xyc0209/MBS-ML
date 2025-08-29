import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
sizes = np.array([50, 100, 75, 125, 200])
labels = ['A', 'B', 'C', 'D', 'E']

# 绘制气泡图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, s=sizes, c='lightblue')

# 添加标注到圆圈内部
for i, label in enumerate(labels):
    plt.text(x[i], y[i], label, ha='center', va='center', fontsize=10)

# 添加网格线
plt.grid(True, which='both')

# 设置标题和坐标轴标签
plt.title('Bubble Chart with Light Blue Bubbles')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()