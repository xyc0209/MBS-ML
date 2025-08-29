import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models
model = joblib.load('../bloated_randomForest_smote.pkl')
bottleneck_model = joblib.load('../bottleneck_randomForest_smote.pkl')
chain_model = joblib.load('../chain_randomForest_smote.pkl')
chatty_model = joblib.load('../chatty_randomForest_smote.pkl')
greedy_model = joblib.load('../greedy_randomForest_smote.pkl')
mega_model = joblib.load('../mega_randomForest_smote.pkl')
nano_model = joblib.load('../nano_randomForest_smote.pkl')


@app.route('/prediction', methods=['POST'])
def predict():
    # Get the source code path from the request body
    request_data = request.get_json()
    reposPath = request_data.get('reposPath')
    outputPath = request_data.get('outputPath')
    if not reposPath:
        return jsonify({'error': 'Source code path is required'}), 400

    # Send a request to localhost:8080/data/features
    response = requests.post('http://localhost:8080/data/features', json={'reposPath': reposPath, 'outputPath': outputPath})

    if response.status_code != 200:
        return jsonify({'error': 'Failed to extract data'}), 500

    # Get the file path from the response
    file_path = response.text

    # Use the loaded models to make predictions
    data_test = pd.read_csv(file_path)
    X_test = data_test.iloc[:, 3:]

    # Use the loaded models to make predictions
    y_pred = model.predict(X_test)
    bottleneck_pred = bottleneck_model.predict(X_test)
    chain_pred = chain_model.predict(X_test)
    chatty_pred = chatty_model.predict(X_test)
    greedy_pred = greedy_model.predict(X_test)
    mega_pred = mega_model.predict(X_test)
    nano_pred = nano_model.predict(X_test)

    # Add the prediction results as new columns
    data_test['Bloated'] = y_pred
    data_test['Bottleneck'] = bottleneck_pred
    data_test['Chain'] = chain_pred
    data_test['Chatty'] = chatty_pred
    data_test['Greedy'] = greedy_pred
    data_test['Mega'] = mega_pred
    data_test['Nano'] = nano_pred

    # Save the updated data back to the file
    data_test.to_csv(file_path, index=False)

    # Return a success response
    return jsonify({'message': '预测结果已写入文件', 'file_path': file_path})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
