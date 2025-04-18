import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/greedy.csv')

X = data.iloc[:, 3:].drop('greedy', axis=1)
y = data['greedy']
scaler = MinMaxScaler()
# 对特征数据进行归一化
X_normalized = scaler.fit_transform(X)

# 将归一化后的数据转换为DataFrame
X = pd.DataFrame(X_normalized, columns=X.columns)
# 使用 SMOTE 进行上采样
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
param_grid = {
    'hidden_layer_sizes': [(20,), (50,), (100,)],  # 隐藏层神经元数量
    'activation': ['relu', 'tanh'],  # 激活函数
    'solver': ['adam', 'sgd'],  # 优化算法
    'alpha': [0.0001, 0.001, 0.01]  # 正则化项的强度
}


# 创建MLP模型
model = MLPClassifier()

# 使用网格搜索选择最佳参数
scoring = ['precision', 'recall', 'f1', 'roc_auc']
grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, refit='precision')
grid_search.fit(X_resampled, y_resampled)

print("Best Parameters:", grid_search.best_params_)
best_model = MLPClassifier(**grid_search.best_params_)

# 进行五折交叉验证
cv_results = cross_validate(best_model, X_resampled, y_resampled, cv=5, scoring=scoring)
best_model.fit(X_resampled, y_resampled)
joblib.dump(best_model, 'greedy_mlp_model.pkl')
# 输出每次交叉验证的准确率、召回率和 F1 值
for i in range(5):
    print(f"Fold {i+1}:")
    print("precision:", cv_results['test_precision'][i])
    print("Recall:", cv_results['test_recall'][i])
    print("F1:", cv_results['test_f1'][i])
    print("AUC:", cv_results['test_roc_auc'][i])
    print()

# 输出五次交叉验证的平均值
print("Average:")
print("precision:", cv_results['test_precision'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1:", cv_results['test_f1'].mean())
print("AUC:", cv_results['test_roc_auc'].mean())

results_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(5)],
    'precision': cv_results['test_precision'],
    'Recall': cv_results['test_recall'],
    'F1': cv_results['test_f1'],
    'AUC': cv_results['test_roc_auc']
})

# 添加平均值行
mean_row = pd.DataFrame({
    'Fold': ['Average'],
    'precision': cv_results['test_precision'].mean(),
    'Recall': cv_results['test_recall'].mean(),
    'F1': cv_results['test_f1'].mean(),
    'AUC': cv_results['test_roc_auc'].mean()
})
results_df = results_df.append(mean_row, ignore_index=True)

# 将结果保存到Excel文件
results_df.to_excel('greedy-mlp-smote-results.xlsx', index=False)