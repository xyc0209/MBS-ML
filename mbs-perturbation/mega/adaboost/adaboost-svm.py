import pandas as pd
from sklearn.metrics import make_scorer, recall_score, accuracy_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

# 读取CSV文件
data = pd.read_csv('/Users/yongchaoxing/desktop/data/mega.csv')

X = data.iloc[:, 3:].drop('Mega', axis=1)
y = data['Mega']


scoring = ['precision', 'recall', 'f1']

# 定义超参数网格
param_grid = {
    'n_estimators': [50, 100, 200],  # AdaBoost 迭代次数
    'learning_rate': [0.1, 1.0, 10.0]  # 学习率
}

# 创建基础分类器
base_classifier = SVC(kernel='linear')

# 创建 AdaBoost 分类器
model = AdaBoostClassifier(base_classifier, algorithm='SAMME')

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, refit='precision')

# 执行网格搜索
grid_search.fit(X, y)

# 输出最佳超参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数组合创建最佳模型
best_model = AdaBoostClassifier(base_classifier, algorithm='SAMME',  **grid_search.best_params_)

# 进行五折交叉验证并计算指标
cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)

# 输出每次交叉验证的准确率、召回率和 F1 值
for i in range(5):
    print(f"Fold {i+1}:")
    print("Accuracy:", cv_results['test_accuracy'][i])
    print("Recall:", cv_results['test_recall'][i])
    print("F1:", cv_results['test_f1'][i])
    print()

# 输出五次交叉验证的平均值
print("Average:")
print("Accuracy:", cv_results['test_accuracy'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1:", cv_results['test_f1'].mean())

results_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(5)],
    'Accuracy': cv_results['test_accuracy'],
    'Recall': cv_results['test_recall'],
    'F1': cv_results['test_f1']
})

# 添加平均值行
mean_row = pd.DataFrame({
    'Fold': ['Average'],
    'Accuracy': cv_results['test_accuracy'].mean(),
    'Recall': cv_results['test_recall'].mean(),
    'F1': cv_results['test_f1'].mean()
})
results_df = results_df.append(mean_row, ignore_index=True)

# 将结果保存到Excel文件
results_df.to_excel('mega-adaboost-svm-results.xlsx', index=False)