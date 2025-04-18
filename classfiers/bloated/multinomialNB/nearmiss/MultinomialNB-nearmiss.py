import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/bloated.csv')

X = data.iloc[:, 3:].drop('bloated', axis=1)
y = data['bloated']





# 定义自定义评分函数
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro'),
    'roc_auc': make_scorer(roc_auc_score)
}

# 使用 NearMiss-1 进行下采样
nm = NearMiss(version=1)
X_resampled, y_resampled = nm.fit_resample(X, y)
# 定义超参数网格
param_grid = {
    'alpha': [0.1, 1.0, 10.0]  # 示例：设置 alpha 超参数的不同取值
}

# 创建决策树模型
model = MultinomialNB()

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, refit='precision')

# 执行网格搜索
grid_search.fit(X_resampled, y_resampled)

# 输出最佳超参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数组合创建最佳模型
best_model = MultinomialNB(**grid_search.best_params_)

# 进行五折交叉验证并计算指标
cv_results = cross_validate(best_model, X_resampled, y_resampled, cv=5, scoring=scoring)
best_model.fit(X_resampled, y_resampled)
joblib.dump(best_model, 'bloated_multinomialNB_nearmiss.pkl')
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
results_df.to_excel('bloated-multinomialNB-nearmiss-results.xlsx', index=False)