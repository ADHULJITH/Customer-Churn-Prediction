#  Customer Churn Prediction System

##  Project Overview

This project builds a Machine Learning-based Customer Churn Prediction System that identifies customers who are likely to leave a service.

By predicting churn in advance, businesses can take proactive retention actions and reduce revenue loss.

The project includes:

* Data preprocessing
* Model training using LightGBM
* Handling class imbalance
* Hyperparameter tuning
* Model evaluation
* Model export for deployment
* Flask-based web application for real-time predictions

## Problem Statement
Customer churn significantly impacts business revenue.
The objective of this project is to:
* Predict whether a customer will churn or not churn.
* Improve detection of churn customers
* Maintain a balanced trade-off between precision and recall
* Deploy the trained model into a web application

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* LightGBM
* Matplotlib, Seaborn
* Flask (Web Application)
* Joblib (Model Saving)
* HTML/CSS (Frontend)
* 
## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                         # Flask application
├── templates/
│   └── index.html                 # Frontend UI
│
├── lightgbm_churn_model.pkl       # Final trained model
├── label_encoders.pkl             # Label encoders
├── columns.pkl                    # Training column order
├── threshold.pkl                  # Prediction threshold
├── requirements.txt               # Project dependencies
└── README.mdd
```

---

##  Model Development

### 1️ Data Preprocessing

* Removed unnecessary columns
* Applied label encoding for categorical features
* Handled class imbalance using `scale_pos_weight`

### 2️ Model Used

LightGBM Classifier

Why LightGBM?

* Handles large datasets efficiently
* Works well with imbalanced data
* High performance and fast training

### 3️ Train-Test Split

* 80% Training
* 20% Testing
* Stratified split to preserve class distribution

---

##  Model Performance (Final Selected Model - Baseline)

| Metric            | Value     |
| ----------------- | --------- |
| Test Accuracy     | **73.6%** |
| ROC-AUC Score     | **0.818** |
| Churn Recall      | **76%**   |
| Weighted F1 Score | **0.75**  |

### Key Observations:

* Strong ability to distinguish churn vs non-churn customers
* High recall for churn customers (important for retention strategy)
* Balanced trade-off between precision and recall
* Stable performance on unseen data

---

##  Hyperparameter Tuning

* Used GridSearchCV with Stratified K-Fold (5 folds)
* Optimized based on ROC-AUC score
* Tuned model improved recall but reduced overall accuracy
* Baseline model was selected as final due to better balance

---

##  Web Application (Flask)

A user-friendly web interface was developed using Flask.

### Features:

* Input customer details through a form
* Real-time churn prediction
* Uses saved model artifacts
* Applies same preprocessing pipeline as training
* Uses optimized decision threshold

### How It Works:

1. User enters customer details
2. Data is encoded using saved label encoders
3. Model predicts churn probability
4. Threshold applied
5. Result displayed instantly

---

##  Model Artifacts Saved

* `lightgbm_churn_model.pkl`
* `label_encoders.pkl`
* `columns.pkl`
* `threshold.pkl`

These ensure:

* Consistent preprocessing
* Correct feature order
* Same prediction logic in production

---

##  How to Run the Project Locally

### 1️ Clone the Repository

```bash
git clone https://github.com/ADHULJITH/Customer-Churn-Prediction.git
cd Customer Churn Prediction
```

### 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️ Run Flask App

```bash
python app.py
```

### 4️ Open in Browser

```
http://127.0.0.1:5000/
```

---

##  Business Impact

This system helps businesses:

* Identify high-risk customers
* Reduce churn rate
* Improve customer retention strategies
* Save marketing costs
* Increase lifetime customer value

---

##  Final Conclusion

The final LightGBM model provides:

* Strong class discrimination
* Reliable churn detection
* Balanced performance metrics
* Production-ready deployment

The model and web application are fully implemented and ready for real-world usage.

---

##  Author

Adhuljith T J

Machine Learning & Data Science Enthusiast

---
