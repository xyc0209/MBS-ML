# MBS-ML (Complex Microservice Bad Smell Detection)

This project provides datasets, training scripts, pre-trained models, and an automated detection tool (*MBS_Detector*) for detecting **complex microservice bad smells (MBSs)**.

------

## 📂 Project Structure

- **/data**
  - Contains training datasets for different types of complex MBSs.
  - `/test` subdirectory provides the complete test set.
- **/classifiers**
  - Organized by different MBS types.
  - Each subdirectory contains:
    - Training scripts for 45 detection models.
    - Evaluation results (Precision, Recall, F1 Score, AUC).
  - `/sota` subdirectory:
    - Pre-trained weights of the best-performing (SOTA) models for each MBS type.
    - Testing scripts for different SOTA models.
    - Source code of the automated detection tool *MBS_Detector* (Flask-based).

------

## 🚀 MBS_Detector Usage

1. **Prepare Metric Extraction Tool**

   - Download and deploy our previously released tool [BSStaticAnalysis](https://github.com/yang66-hash/BSStaticAnalysis).
   - Send a POST request to the `/data/features` API to extract microservice metric information:

   ```
   {  
       "reposPath": "path/to/microservice/source",  
       "outputPath": "path/to/output.csv"  
   }
   ```

2. **Configure MBS_Detector**

   - Open `/classifiers/sota/MBS_Detector/MBS_Detector.py`.
   - At line 28, update the `requests.post` URL to match the actual IP:PORT of the deployed **BSStaticAnalysis** service (default: `http://localhost:8080`).
   - Example:

   ```
   response = requests.post('http://<ip>:<port>/data/features', 
                            json={'reposPath': reposPath, 'outputPath': outputPath})
   ```

3. **Start the Tool**

   - Run the following command to launch *MBS_Detector*:

   ```
   python MBS_Detector.py
   ```

4. **Run Detection**

   - Send a POST request to the `prediction` API of *MBS_Detector* with the following request body:

   ```
   {  
       "reposPath": "path/to/microservice/source",  
       "outputPath": "path/to/detection_result.csv"  
   }
   ```

   - The detection results will be written to the specified output file.

5. **Interpret Results**

   - Based on the detection output, users can decide whether to refactor or adjust the microservice system to remove detected bad smells.
------
## 📌 Notes

- Ensure required dependencies (`Flask`, `pandas`, `requests`, `joblib`) are installed.
- Confirm the BSStaticAnalysis service is running on the specified IP:port before starting detection.