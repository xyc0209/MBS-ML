import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score



# 加载模型
model = joblib.load('chatty_randomForest_smote.pkl')
print(model)
# 读取测试数据
data_test = pd.read_csv('D:/dataSet/mbs-interference/chatty-test.csv')
X_test = data_test.iloc[:, 0:].drop('chatty', axis=1)
y_test = data_test['chatty']

# 使用加载的模型进行预测
y_pred = model.predict(X_test)

# 计算precision和recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('precision:', precision)
print('recall:', recall)