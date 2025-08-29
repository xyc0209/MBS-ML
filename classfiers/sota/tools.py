import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score



# 加载模型
model = joblib.load('bloated_randomForest_smote.pkl')
bottleneck_model = joblib.load('bottleneck_randomForest_smote.pkl')
chain_model = joblib.load('chain_randomForest_smote.pkl')
chatty_model = joblib.load('chatty_randomForest_smote.pkl')
greedy_model = joblib.load('greedy_randomForest_smote.pkl')
mega_model = joblib.load('mega_randomForest_smote.pkl')
nano_model = joblib.load('nano_randomForest_smote.pkl')
print(model)
# 读取测试数据
data_test = pd.read_csv('D:/work_space/PropertyManagementCloud-master-data.csv')
X_test = data_test.iloc[:, 3:]

# 使用加载的模型进行预测
y_pred = model.predict(X_test)
bottleneck_pred = bottleneck_model.predict(X_test)
chain_pred = chain_model.predict(X_test)
chatty_pred = chatty_model.predict(X_test)
greedy_pred = greedy_model.predict(X_test)
mega_pred = mega_model.predict(X_test)
nano_pred = nano_model.predict(X_test)

# 将预测结果作为新的列添加
data_test['Bloated'] = y_pred
data_test['Bottleneck'] = bottleneck_pred
data_test['Chain'] = chain_pred
data_test['Chatty'] = chatty_pred
data_test['Greedy'] = greedy_pred
data_test['Mega'] = mega_pred
data_test['Nano'] = nano_pred

# 将更新后的数据保存回文件
data_test.to_csv('D:/work_space/PropertyManagementCloud-master-data.csv', index=False)

print("预测结果已写入 'bloated-test.csv'.")