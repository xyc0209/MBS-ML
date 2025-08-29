import pandas as pd
from sklearn.metrics import make_scorer, recall_score, accuracy_score, f1_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

# 读取CSV文件
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('/Users/yongchaoxing/desktop/data/mega.csv')

X = data.iloc[:, 3:].drop('Mega', axis=1)
y = data['Mega']

# 定义自定义评分函数
scoring = ['accuracy', 'recall', 'f1']

# 定义决策树的超参数网格
param_grid_deciosnTree = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建基础分类器
base_classifier = DecisionTreeClassifier()

# 创建 GridSearchCV 对象
rf_grid_search = GridSearchCV(base_classifier, param_grid_deciosnTree, scoring=scoring, cv=5, refit='accuracy')

# 执行随机森林的网格搜索
rf_grid_search.fit(X, y)

# 输出随机森林最佳超参数组合
print("DecisionTree Best Parameters:", rf_grid_search.best_params_)


# 定义 AdaBoost 的超参数网格
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],  # AdaBoost 迭代次数
    'learning_rate': [0.1, 1.0, 10.0]  # 学习率
}

# 使用随机森林最佳超参数组合创建最佳基础分类器
best_base_classifier = DecisionTreeClassifier(**rf_grid_search.best_params_)
# 创建 AdaBoost 分类器
model = AdaBoostClassifier(best_base_classifier)
# 创建 AdaBoost 的 GridSearchCV 对象
grid_search = GridSearchCV(model, param_grid_adaboost, scoring=scoring, cv=5, refit='accuracy')

# 执行 AdaBoost 的网格搜索
grid_search.fit(X, y)

# 输出 AdaBoost 最佳超参数组合
print("AdaBoost Best Parameters:", grid_search.best_params_)

# 使用最佳超参数组合创建最佳模型
best_model = AdaBoostClassifier(best_base_classifier, **grid_search.best_params_)

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
results_df.to_excel('mega-adaboost-decisionTree-results.xlsx', index=False)