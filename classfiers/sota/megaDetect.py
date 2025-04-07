import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score

# 加载模型
model = joblib.load('mega_randomForest_smote.pkl')
print(model)
# 读取测试数据
data_test = pd.read_csv('D:/dataSet/mbs/mega-test.csv')
X_test = data_test.iloc[:, 0:].drop('Mega', axis=1)
y_test = data_test['Mega']
print(X_test.columns)


# 使用加载的模型进行预测
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('precision:', precision)
print('recall:', recall)
print('accuracy:', accuracy)

importance = model.feature_importances_
feature_names = X_test.columns

# 创建一个包含特征重要性和特征名称的DataFrame
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
# 根据特征重要性降序排序
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

print(feature_importance_df)