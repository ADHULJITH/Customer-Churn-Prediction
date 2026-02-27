# Customer Churn Prediction App

A Flask web application that predicts customer churn using a trained LightGBM model.

---

## 📁 Project Structure

```
Customer churn prediction/
├── app.py                        ← Flask backend
├── requirements.txt              ← Python dependencies
├── templates/
│   └── index.html                ← Frontend UI
├── lightgbm_churn_model.pkl      ← Trained model  ← PLACE HERE
├── label_encoders.pkl            ← Label encoders ← PLACE HERE
├── columns.pkl                   ← Column order   ← PLACE HERE
└── threshold.pkl                 ← Decision threshold ← PLACE HERE
```

---

## 🚀 Setup & Run

### 1. Place your .pkl files
Copy your 4 model artifact files into the `churn-app/` root folder:
- `lightgbm_churn_model.pkl`
- `label_encoders.pkl`
- `columns.pkl`
- `threshold.pkl`

### 2. Install dependencies
```bash
cd Customer churn prediction
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🎯 How It Works

1. User fills in customer details in the form
2. On submit, the form sends a POST request to `/predict`
3. Flask loads the pkl artifacts, preprocesses input, and runs prediction
4. The result (churn probability + risk level) is shown with an animated gauge

---

## 📊 Input Fields

| Field | Type | Description |
|-------|------|-------------|
| gender | Categorical | Male / Female |
| SeniorCitizen | Binary | 0 / 1 |
| Partner | Categorical | Yes / No |
| Dependents | Categorical | Yes / No |
| tenure | Numeric | Months with company |
| Contract | Categorical | Month-to-month / One year / Two year |
| PaperlessBilling | Categorical | Yes / No |
| PaymentMethod | Categorical | 4 options |
| MonthlyCharges | Numeric | Monthly bill amount |
| TotalCharges | Numeric | Cumulative billing |
| PhoneService | Categorical | Yes / No |
| MultipleLines | Categorical | Yes / No / No phone service |
| InternetService | Categorical | Fiber optic / DSL / No |
| OnlineSecurity | Categorical | Yes / No / No internet service |
| OnlineBackup | Categorical | Yes / No / No internet service |
| DeviceProtection | Categorical | Yes / No / No internet service |
| TechSupport | Categorical | Yes / No / No internet service |
| StreamingTV | Categorical | Yes / No / No internet service |
| StreamingMovies | Categorical | Yes / No / No internet service |

