import numpy as np
import pandas as pd
from hpelm import ELM
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import load_iris
from sklearn.utils import compute_sample_weight


class WELM(ELM):
    def __init__(self, inputs, outputs, weighted=True):
        super().__init__(inputs=inputs, outputs=outputs)
        self.weighted = weighted

    def train(self, X, y, sample_weights=None):
        if self.weighted and sample_weights is not None:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = None
        super().train(X, y)

    def _train(self, X, t):
        if self.sample_weights is not None:
            self.wi = self.sample_weights
        super()._train(X, t)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int)  # 将概率值转换为二进制类别
    accuracy = accuracy_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    auc = roc_auc_score(y, y_pred)  # 不需要指定 average 和 multi_class 参数
    return accuracy, recall, f1, auc

# Load dataset (example: Iris)
data = pd.read_csv('D:/dataSet/mbs/nano.csv')

X = data.iloc[:, 3:].drop('nano', axis=1)
y = data['nano']
smote = SMOTE()
X, y = smote.fit_resample(X, y)
# Calculate class weights
class_weights = compute_sample_weight('balanced', y)
# Initialize list to store results
results = []

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Initialize and train model
    model = WELM(X_train.shape[1], 1, weighted=True)
    # Set sample weights
    sample_weights = class_weights[train_idx]  # 根据训练样本的索引设置样本权重
    model.add_neurons(30, 'rbf_linf')
    X_train_array = X_train.to_numpy()
    y_train_array = y_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_test_array = y_test.to_numpy()
    model.train(X_train_array, y_train_array, sample_weights=sample_weights)
    print("sample_weights",sample_weights)
    # Evaluate model
    accuracy, recall, f1, auc = evaluate_model(model, X_test_array, y_test_array)

    # Append results to list
    results.append({'Fold': fold, 'Accuracy': accuracy, 'Recall': recall,
                    'F1': f1, 'AUC': auc})
    print("fold:", fold)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc)
# Convert results list to DataFrame
output = pd.DataFrame(results)
print("Average")
print('Accuracy',output['Accuracy'].mean())
print('Recall',output['Recall'].mean())
print('F1',output['F1'].mean())
print('AUC',output['AUC'].mean())
# Calculate average of evaluation metrics
average_results = output.mean()
# 添加平均值行
mean_row = pd.DataFrame({
    'Fold': ['Average'],
    'Accuracy': output['Accuracy'].mean(),
    'Recall': output['Recall'].mean(),
    'F1': output['F1'].mean(),
    'AUC': output['AUC'].mean()
})
output = output.append(mean_row, ignore_index=True)



# Save results to Excel
output.to_excel('nano_welm_rbf_smote_results.xlsx', index=False)
