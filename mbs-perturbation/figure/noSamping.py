import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn的配色方案
sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)

[0.916678322, 1, 1, 1, 0.831449275, 0.95057097, 0.985717816]
# 创建每个子图的分类器性能数据
classifier_scores_list = [
    [[0.9473, 0.9843, 0.9316, 0.9778, 0.9944, 0.9806, 0.9429]],  # MS的数据
    [[0.9991, 0.997, 0.9793, 1, 0.9958, 0.9993, 0.9997]],  # NS的数据
    [[0.6999, 0.6883, 0.6981, 0.6729, 0.6481, 0.57, 0.6656]],  # BS的数据
    [[0.7222, 0.7056, 0.7147, 0.7551, 0.7062, 0.6721, 0.6912]],  # CS的数据
    [[0.9325, 0.9232, 0.9036, 0.9732, 0.8327, 0.8995, 0.9044]],  # BOS的数据
    [[0.9864, 0.9964, 0.9605, 0.9982, 0.8698, 0.9569, 0.9889]],  # SC的数据
    [[0.776, 0.7215, 0.7069, 0.8638, 0.6184, 0.7593, 0.814]],  # MG的数据
]

# 创建每个子图的分类器标签
classifiers_list = [
    ['DT'],  # 子图1的分类器
    ['RF'],  # 子图2的分类器
    ['ELM'],  # 子图3的分类器
    ['WELM'],  # 子图4的分类器
    ['KNN'],  # 子图5的分类器
    ['MLP'],
    ['MNB'],
]

# 创建2行4列的网格布局
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

# 在每个子图中绘制箱线图
for i, ax in enumerate(axes.flatten()):
    classifier_scores = classifier_scores_list[i]
    classifiers = classifiers_list[i]
    ax.boxplot(classifier_scores)
    ax.set_title(classifiers[0])
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()