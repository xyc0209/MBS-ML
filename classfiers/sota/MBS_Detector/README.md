# Automated Complex MBSs Detection Tool — *MBS_Detector*

This tool integrates both the metric extraction process and the detection process based on state-of-the-art (SOTA) machine learning models.

### Usage Steps:

1. **Start BSStaticAnalysis**

   Ensure that the API at `localhost:8080/data/features` is accessible for metric extraction.

2. **Start the Flask Service**

   ```python
   python MBS_Detector.py
   ```

3. **Send Detection Request via Postman or curl**

```json
POST http://localhost:5000/prediction
Content-Type: application/json

{
  "reposPath": "Path to your source code",
  "outputPath": "Path to save the detection results"
}
```

4. **Successful Response**

If the detection is successful, the following response will be returned:

```json
{
  "message": "Prediction results have been written to the file",
  "file_path": "Path to the generated detection result CSV file"
}

```

