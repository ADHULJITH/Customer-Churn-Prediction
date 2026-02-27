from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model artifacts (no threshold.pkl needed)
model = joblib.load("lightgbm_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
training_columns = joblib.load("columns.pkl")

# ── Threshold Configuration ────────────────────────────────────────────────
# Using 0.35 instead of 0.40 to be MORE sensitive to churn detection.
# Lower threshold = catches more churners (fewer false negatives).
# If your model still under-predicts churn, lower this further (e.g. 0.30).
THRESHOLD = 0.20

@app.route("/")
def index():
    return render_template("index.html", threshold=round(THRESHOLD * 100))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Build DataFrame ────────────────────────────────────────────────
        input_df = pd.DataFrame([data])

        # ── Label-encode all categorical columns ───────────────────────────
        for col, le in label_encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0]).strip()
                if val in le.classes_:
                    input_df[col] = int(le.transform([val])[0])
                else:
                    # Unknown value → default to 0
                    print(f"[WARN] Unknown value '{val}' for column '{col}'. Defaulting to 0.")
                    input_df[col] = 0

        # ── Align columns to training schema ──────────────────────────────
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[training_columns]

        # ── Final numeric safety sweep ─────────────────────────────────────
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # ── Debug print ───────────────────────────────────────────────────
        print("\n[DEBUG] Input to model:")
        print(input_df.dtypes)
        print(input_df)

        # ── Predict ────────────────────────────────────────────────────────
        prob = float(model.predict_proba(input_df)[:, 1][0])
        prediction = int(prob >= THRESHOLD)

        print(f"[DEBUG] Probability: {prob:.4f} | Threshold: {THRESHOLD} | Prediction: {'CHURN' if prediction else 'STAY'}")

        # ── Risk level ─────────────────────────────────────────────────────
        if prob >= 0.65:
            risk_level, risk_color = "High",   "#ef4444"
        elif prob >= 0.35:
            risk_level, risk_color = "Medium", "#f59e0b"
        else:
            risk_level, risk_color = "Low",    "#10b981"

        return jsonify({
            "success":     True,
            "churn":       prediction,
            "probability": round(prob * 100, 2),
            "risk_level":  risk_level,
            "risk_color":  risk_color,
            "threshold":   round(THRESHOLD * 100),
            "message":     "This customer is likely to churn." if prediction else "This customer is likely to stay."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
