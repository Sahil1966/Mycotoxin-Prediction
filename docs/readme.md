# 🌽 Mycotoxin Prediction in Corn  

## 📌 Overview  
This project aims to predict deoxynivalenol (DON) concentration in corn using hyperspectral imaging data.  
The pipeline includes data preprocessing, feature engineering, model training, evaluation, and deployment via a Streamlit web app.

---

## 🚀 Setup Instructions  

### Clone the Repository  
```bash
git clone https://github.com/Sahil1966/Mycotoxin-Prediction.git
cd mycotoxin-prediction

### Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

### Install Dependencies
pip install -r requirements.txt

### Prepare the Data
mycotoxin-prediction/
│── data/
│   ├── processed_data.csv

### 📂 Project Structure
mycotoxin-prediction
│── data/                   # Stores raw and processed datasets
│── notebooks/              # Jupyter notebooks for EDA, preprocessing
│── src/                    # Source code for the ML pipeline
│   ├── data_preprocessing/ # Data loading, cleaning, feature engineering
│   ├── models/             # Model training, evaluation, and tuning
│   ├── utils/              # Logging, helper functions
│── tests/                  # Unit tests for validation
│── deployment/             # API & Streamlit app for model deployment
│── logs/                   # Log files for debugging
│── requirements.txt        # Dependency list
│── README.md               # Documentation
│── setup.py                # Package setup

### 🎯 Running the Pipeline

### Train the Model
python src/models/train.py

### Hyperparameter Tuning
python src/models/tuning.py

### Run Streamlit App
streamlit run deployment/app.py

### 📊 Model Evaluation
R² Score: -0.269 (Model struggling to capture patterns)
RMSE: 885.023 (Large prediction errors)
Feature Importance (SHAP Analysis):
Some features have strong predictive power (e.g., 78, 123, 136).
Model struggles with extreme values & generalization.

### 🔥 Future Improvements
✅ Improve feature selection & dimensionality reduction
✅ Try different model architectures (XGBoost, CNNs for spectral data)
✅ Handle outliers & imbalanced data better
✅ Deploy using Docker & cloud services



