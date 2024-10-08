import joblib
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/nano.csv')

X = data.iloc[:, 3:].drop('nano', axis=1)
y = data['nano']
scaler = MinMaxScaler()
# 对特征数据进行归一化
X_normalized = scaler.fit_transform(X)

# 将归一化后的数据转换为DataFrame
X = pd.DataFrame(X_normalized, columns=X.columns)
# 使用 NearMiss-1 进行下采样
nm = NearMiss(version=1)
X_resampled, y_resampled = nm.fit_resample(X, y)
# 定义参数网格
param_grid = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# 创建KNN模型
model = KNeighborsClassifier()
scoring = ['precision', 'recall', 'f1', 'roc_auc']

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, refit='precision')
grid_search.fit(X_resampled, y_resampled)

# 输出最佳超参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数组合创建KNN模型
best_model = KNeighborsClassifier(**grid_search.best_params_)

# 进行五折交叉验证并计算指标
cv_results = cross_validate(best_model, X_resampled, y_resampled, cv=5, scoring=scoring)

# 输出每次交叉验证的准确率、召回率和F1值

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
results_df.to_excel('nano-knn-nearmiss-results.xlsx', index=False)