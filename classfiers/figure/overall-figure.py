import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn的配色方案
sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)

# 创建七个分类器性能的示例数据
classifier_scores = [
    [0.9473, 0.9843, 0.9316, 0.9778, 0.9944, 0.9806, 0.9429],
    [0.9991, 0.997, 0.9793, 1, 0.9958, 0.9993, 0.9997],
    [0.6999, 0.6883, 0.6981, 0.6729, 0.6481, 0.57, 0.6656],
    [0.7222, 0.7056, 0.7147, 0.7551, 0.7062, 0.6721, 0.6912],
    [0.9325, 0.9232, 0.9036, 0.9732, 0.8327, 0.8995, 0.9044],
    [0.9864, 0.9964, 0.9605, 0.9982, 0.8698, 0.9569, 0.9889],
    [0.776, 0.7215, 0.7069, 0.8638, 0.6184, 0.7593, 0.814],
    [0.8375, 0.9045, 0.7881, 0.7054, 0.6549, 0.6379, 0.8375]
]

# 绘制箱线图
plt.boxplot(classifier_scores, labels=['DT', 'RF', 'ELM', 'WELM', 'KNN', 'MLP', 'MNB', 'SVM'])
plt.title('Classifier Performance')
plt.xlabel('Classifiers')
plt.ylabel('AUC')
plt.tight_layout()
output_file = "./overall-auc.jpg"
plt.savefig(output_file, format='jpeg', dpi=400)  # 设置dpi参数可以调整图像
plt.show()