from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Numerical Inputs ──────────────────────────────────
        senior_citizen   = int(request.form["SeniorCitizen"])
        monthly_charges  = float(request.form["MonthlyCharges"])
        total_charges    = float(request.form["TotalCharges"])
        tenure           = int(request.form["tenure"])

        # ── Categorical Inputs ────────────────────────────────
        gender           = request.form["gender"]
        partner          = request.form["Partner"]
        dependents       = request.form["Dependents"]
        phone_service    = request.form["PhoneService"]
        multiple_lines   = request.form["MultipleLines"]
        internet_service = request.form["InternetService"]
        online_security  = request.form["OnlineSecurity"]
        online_backup    = request.form["OnlineBackup"]
        device_protect   = request.form["DeviceProtection"]
        tech_support     = request.form["TechSupport"]
        streaming_tv     = request.form["StreamingTV"]
        streaming_movies = request.form["StreamingMovies"]
        contract         = request.form["Contract"]
        paperless        = request.form["PaperlessBilling"]
        payment          = request.form["PaymentMethod"]

        # ── Tenure Group ──────────────────────────────────────
        if   tenure <= 12: tenure_group = "1 - 12"
        elif tenure <= 24: tenure_group = "13 - 24"
        elif tenure <= 36: tenure_group = "25 - 36"
        elif tenure <= 48: tenure_group = "37 - 48"
        elif tenure <= 60: tenure_group = "49 - 60"
        else:              tenure_group = "61 - 72"

        # ── Build Feature Dict (all 50 features) ─────────────
        features = {
            "SeniorCitizen": senior_citizen,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,

            "gender_Female": 1 if gender == "Female" else 0,
            "gender_Male":   1 if gender == "Male"   else 0,

            "Partner_No":  1 if partner == "No"  else 0,
            "Partner_Yes": 1 if partner == "Yes" else 0,

            "Dependents_No":  1 if dependents == "No"  else 0,
            "Dependents_Yes": 1 if dependents == "Yes" else 0,

            "PhoneService_No":  1 if phone_service == "No"  else 0,
            "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

            "MultipleLines_No":                 1 if multiple_lines == "No"                 else 0,
            "MultipleLines_No phone service":   1 if multiple_lines == "No phone service"   else 0,
            "MultipleLines_Yes":                1 if multiple_lines == "Yes"                else 0,

            "InternetService_DSL":          1 if internet_service == "DSL"          else 0,
            "InternetService_Fiber optic":  1 if internet_service == "Fiber optic"  else 0,
            "InternetService_No":           1 if internet_service == "No"           else 0,

            "OnlineSecurity_No":                   1 if online_security == "No"                   else 0,
            "OnlineSecurity_No internet service":  1 if online_security == "No internet service"  else 0,
            "OnlineSecurity_Yes":                  1 if online_security == "Yes"                  else 0,

            "OnlineBackup_No":                   1 if online_backup == "No"                   else 0,
            "OnlineBackup_No internet service":  1 if online_backup == "No internet service"  else 0,
            "OnlineBackup_Yes":                  1 if online_backup == "Yes"                  else 0,

            "DeviceProtection_No":                   1 if device_protect == "No"                   else 0,
            "DeviceProtection_No internet service":  1 if device_protect == "No internet service"  else 0,
            "DeviceProtection_Yes":                  1 if device_protect == "Yes"                  else 0,

            "TechSupport_No":                   1 if tech_support == "No"                   else 0,
            "TechSupport_No internet service":  1 if tech_support == "No internet service"  else 0,
            "TechSupport_Yes":                  1 if tech_support == "Yes"                  else 0,

            "StreamingTV_No":                   1 if streaming_tv == "No"                   else 0,
            "StreamingTV_No internet service":  1 if streaming_tv == "No internet service"  else 0,
            "StreamingTV_Yes":                  1 if streaming_tv == "Yes"                  else 0,

            "StreamingMovies_No":                   1 if streaming_movies == "No"                   else 0,
            "StreamingMovies_No internet service":  1 if streaming_movies == "No internet service"  else 0,
            "StreamingMovies_Yes":                  1 if streaming_movies == "Yes"                  else 0,

            "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
            "Contract_One year":       1 if contract == "One year"       else 0,
            "Contract_Two year":       1 if contract == "Two year"       else 0,

            "PaperlessBilling_No":  1 if paperless == "No"  else 0,
            "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,

            "PaymentMethod_Bank transfer (automatic)":  1 if payment == "Bank transfer (automatic)"  else 0,
            "PaymentMethod_Credit card (automatic)":    1 if payment == "Credit card (automatic)"    else 0,
            "PaymentMethod_Electronic check":           1 if payment == "Electronic check"           else 0,
            "PaymentMethod_Mailed check":               1 if payment == "Mailed check"               else 0,

            "tenure_group_1 - 12":   1 if tenure_group == "1 - 12"   else 0,
            "tenure_group_13 - 24":  1 if tenure_group == "13 - 24"  else 0,
            "tenure_group_25 - 36":  1 if tenure_group == "25 - 36"  else 0,
            "tenure_group_37 - 48":  1 if tenure_group == "37 - 48"  else 0,
            "tenure_group_49 - 60":  1 if tenure_group == "49 - 60"  else 0,
            "tenure_group_61 - 72":  1 if tenure_group == "61 - 72"  else 0,
        }

        input_df = pd.DataFrame([features])
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        if prediction == 1:
            result     = "⚠️ This customer is likely to CHURN"
            result_class = "danger"
        else:
            result     = "✅ This customer is likely to STAY"
            result_class = "success"

        confidence = f"Churn Probability: {probability:.2f}%"

        return render_template("home.html", result=result,
                               confidence=confidence, result_class=result_class)

    except Exception as e:
        return render_template("home.html", result=f"Error: {str(e)}",
                               result_class="warning", confidence="")

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)