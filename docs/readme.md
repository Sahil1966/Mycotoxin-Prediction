---

# Mycotoxin Prediction in Corn Using Hyperspectral Imaging

## 📌 Overview

This project leverages hyperspectral imaging data to predict deoxynivalenol (DON) concentrations in corn. The workflow includes data preprocessing, feature engineering, model training, evaluation, and deployment via a Streamlit web application.

---

## 🚀 Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/Sahil1966/Mycotoxin-Prediction.git
cd mycotoxin-prediction
```


### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```


### Install Dependencies

```bash
pip install -r requirements.txt
```


### Prepare the Data

Ensure the following structure for your data files:

```
mycotoxin-prediction/
│── data/
│   ├── processed_data.csv
```

---

## 📂 Project Structure

| Folder/File | Description |
| :-- | :-- |
| `data/` | Stores raw and processed datasets |
| `notebooks/` | Jupyter notebooks for EDA and preprocessing |
| `src/` | Source code for the ML pipeline |
| `src/data_preprocessing/` | Data loading, cleaning, and feature engineering |
| `src/models/` | Model training, evaluation, and tuning |
| `src/utils/` | Logging and helper functions |
| `tests/` | Unit tests for validation |
| `deployment/` | API and Streamlit app for model deployment |
| `logs/` | Log files for debugging |
| `requirements.txt` | Dependency list |
| `README.md` | Documentation |
| `setup.py` | Package setup |

---

## 🎯 Running the Pipeline

### Train the Model

```bash
python src/models/train.py
```


### Hyperparameter Tuning

```bash
python src/models/tuning.py
```


### Run Streamlit App

```bash
streamlit run deployment/app.py
```

---

## 📊 Model Evaluation

- **R² Score**: -0.269 (Model struggling to capture patterns)
- **RMSE**: 885.023 (Large prediction errors)

**Feature Importance (via SHAP Analysis):**

- Some features (e.g., 78, 123, 136) exhibit strong predictive power.
- Model struggles with extreme values and generalization.

---

## 🔥 Future Improvements

- ✅ Enhance feature selection and apply dimensionality reduction techniques.
- ✅ Experiment with advanced model architectures like XGBoost or CNNs tailored for spectral data.
- ✅ Address outliers and imbalanced datasets effectively.
- ✅ Deploy the application using Docker and integrate with cloud services for scalability.

---
