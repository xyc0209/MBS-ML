import joblib
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn import svm
from sklearn.model_selection import cross_validate

# 读取CSV文件
data = pd.read_csv('D:/dataSet/mbs/chain.csv')

X = data.iloc[:, 3:].drop('chain', axis=1)
y = data['chain']
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
correlations_df.to_excel('correlations-chain.xlsx', index=False)
