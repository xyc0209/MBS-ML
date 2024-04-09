import joblib
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn import svm
from sklearn.model_selection import cross_validate

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/mega.csv')

X = data.iloc[:, 3:].drop('Mega', axis=1)
y = data['Mega']
# 计算每个特征和标签之间的斯皮尔曼等级相关系数
correlations = {}
for feature in X.columns:
    correlation, p_value = spearmanr(X[feature], y)
    correlations[feature] = correlation

# 打印每个特征和标签的相关系数
for feature, correlation in correlations.items():
    print(f'{feature}: {correlation}')
# correlation_matrix = X.corr()
# print("correlation_matrix",correlation_matrix)
# 保存相关系数矩阵到 Excel 文件
# 保存每个特征和标签的相关性到 Excel 文件
correlations_df = pd.DataFrame({'Feature': correlations.keys(), 'Correlation': correlations.values()})
correlations_df.to_excel('correlations.xlsx', index=False)

# 创建SVM模型
model = svm.SVC(kernel='sigmoid')

# 进行五折交叉验证
scoring = ['precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
# 保存模型
joblib.dump(model, '../mega_svm_sigmoid.pkl')
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
results_df.to_excel('mega-svm-sigmoid-results.xlsx', index=False)