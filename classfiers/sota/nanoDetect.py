import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score,accuracy_score



# 加载模型
model = joblib.load('nano_randomForest_smote.pkl')
print(model)
# 读取测试数据
data_test = pd.read_csv('D:/dataSet/mbs/nano-test.csv')
X_test = data_test.iloc[:, 0:].drop('nano', axis=1)
y_test = data_test['nano']

# 使用加载的模型进行预测
y_pred = model.predict(X_test)

# 计算precision和recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('precision:', precision)
print('recall:', recall)
print('accuracy:', accuracy)

importance = model.feature_importances_
feature_names = X_test.columns
# 计算accuracy

# 创建一个包含特征重要性和特征名称的DataFrame
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
# 根据特征重要性降序排序
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

print(feature_importance_df)