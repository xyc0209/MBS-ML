import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/chatty.csv')

X = data.iloc[:, 3:].drop('chatty', axis=1)
y = data['chatty']
scaler = MinMaxScaler()
# 对特征数据进行归一化
X_normalized = scaler.fit_transform(X)

# 将归一化后的数据转换为DataFrame
X = pd.DataFrame(X_normalized, columns=X.columns)
smote = SMOTE()
X, y = smote.fit_resample(X, y)
# 创建SVM模型
model = svm.SVC(kernel='linear')

# 进行五折交叉验证
scoring = ['precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
model.fit(X, y)
joblib.dump(model, 'chatty_svm_linear_smote.pkl')
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
results_df.to_excel('chatty-svm-linear-results.xlsx', index=False)