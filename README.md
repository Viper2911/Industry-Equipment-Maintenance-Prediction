# ⚙️ Industrial Equipment Maintenance Prediction

An interactive web application and machine learning model designed to forecast potential industrial equipment failures before they happen. This project shifts the maintenance paradigm from reactive (fixing broken machines) to proactive (scheduling repairs based on data-driven failure probabilities).

## 🚀 Overview

This system utilizes historical sensor telemetry (such as temperature differentials, torque, and rotational speed) to predict the likelihood of an imminent machine breakdown. By identifying the specific features contributing to equipment stress, industrial teams can reduce unexpected downtime and optimize equipment reliability.

The model was trained on the AI4I 2020 Predictive Maintenance Dataset and heavily addresses class imbalance using SMOTE to ensure high sensitivity to actual failures.

## 🧠 Key Features

* **Live Telemetry Dashboard:** A Streamlit-powered UI allowing engineers to input live sensor readings and receive instant failure probability scores.
* **Feature Engineering:** Calculates custom physical stress metrics, including cooling strain (temperature differentials) and mechanical power output.
* **XGBoost Classifier:** A robust gradient-boosting architecture tuned for tabular sensor data.
* **Interpretability:** Integrates SHAP (SHapley Additive exPlanations) in the underlying notebook to isolate and display the top features driving the failure predictions.

## 🛠️ Technology Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Explainable AI:** SHAP
* **Data Processing:** Pandas, NumPy

## ⚙️ Installation and Usage

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Viper2911/Industry-Equipment-Maintenance-Prediction.git](https://github.com/Viper2911/Industry-Equipment-Maintenance-Prediction.git)
   cd Industry-Equipment-Maintenance-Prediction
   
2. **Install the required dependencies:**

  ```bash
  pip install -r requirements.txt
