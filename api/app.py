# from flask import Flask, request, render_template, redirect, url_for
# import pandas as pd
# import joblib
# import os
# from src.detect_drift import check_data_drift

# # --- Flask app ---
# app = Flask(__name__, template_folder="templates")

# # --- Config Paths ---
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# MODEL_URI_PATH = os.path.join(BASE_DIR, "latest_model_uri.txt")
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "reference_data.csv")
# NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new_data.csv")

# # --- Load model ---
# def load_model():
#     if not os.path.exists(MODEL_URI_PATH):
#         raise FileNotFoundError(f"latest_model_uri.txt not found at {MODEL_URI_PATH}")
    
#     with open(MODEL_URI_PATH, "r") as f:
#         model_uri = f.read().strip()

#     model_filename = model_uri.split("/")[-1] + ".pkl"
#     model_path = os.path.join(MODEL_DIR, model_filename)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")

#     return joblib.load(model_path)

# # --- Preprocessing ---
# def preprocess_new_data(df):
#     from src.preprocessing import preprocess_data
#     X_processed, _ = preprocess_data(df, training=False)
#     return X_processed

# # --- Home route ---
# @app.route("/")
# def index():
#     return redirect(url_for("predict"))

# # --- Prediction route ---
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         file = request.files.get("file")
#         if not file or not file.filename.endswith(".csv"):
#             return render_template("index.html", error="Please upload a valid CSV file.")

#         try:
#             df = pd.read_csv(file)
#             df.to_csv('data/new_data.csv', index=False)
#             original_df = df.copy()

#             # 🔁 Save new data to Airflow input location
#             data_dir = os.path.join(BASE_DIR, "data")
#             os.makedirs(data_dir, exist_ok=True)
#             df.to_csv(NEW_DATA_PATH, index=False)

#             drift_message = ""

#             # 🔍 Drift Detection (optional)
#             if os.path.exists(REFERENCE_DATA_PATH):
#                 reference_df = pd.read_csv(REFERENCE_DATA_PATH)
#                 is_drift = check_data_drift(reference_df, df)
#                 drift_message = "⚠️ Drift Detected!" if is_drift else "✅ No drift detected"
#             else:
#                 drift_message = "⚠️ Reference data not found, skipping drift detection"
#                 is_drift = False

#             # 🔮 Predict
#             X_processed = preprocess_new_data(df)
#             model = load_model()
#             predictions = model.predict(X_processed)
#             original_df["Prediction"] = predictions

#             print(f"✅ Drift detected: {is_drift}")
#             print(f"📦 Drift message: {drift_message}")
#             return render_template(
#                 "results.html",
#                 tables=[original_df.to_html(classes='table table-bordered table-striped', index=False)],
#                 titles=original_df.columns.values,
#                 drift_message=drift_message
#             )

#         except Exception as e:
#             return render_template("index.html", error=f"❌ Prediction failed: {str(e)}")

#     return render_template("index.html")







# from flask import Flask, request, render_template, redirect, url_for
# import pandas as pd
# import joblib
# import os
# from src.detect_drift import check_data_drift

# # Flask app instance
# app = Flask(__name__, template_folder="templates")

# # --- Config Paths ---
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# MODEL_URI_PATH = os.path.join(BASE_DIR, "latest_model_uri.txt")
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "reference_data.csv")



# # --- Load trained model ---
# def load_model():
#     if not os.path.exists(MODEL_URI_PATH):
#         raise FileNotFoundError(f"latest_model_uri.txt not found at {MODEL_URI_PATH}")
    
#     with open(MODEL_URI_PATH, "r") as f:
#         model_uri = f.read().strip()
    
#     model_filename = model_uri.split("/")[-1] + ".pkl"
#     model_path = os.path.join(MODEL_DIR, model_filename)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")

#     return joblib.load(model_path)

# # --- Preprocess uploaded CSV ---
# def preprocess_new_data(df):
#     from src.preprocessing import preprocess_data
#     X_processed, _ = preprocess_data(df, training=False)
#     return X_processed

# # --- Home route ---
# @app.route("/")
# def index():
#     return redirect(url_for("predict"))

# # --- Prediction route ---
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         file = request.files.get("file")
#         if not file or not file.filename.endswith(".csv"):
#             return render_template("index.html", error="Please upload a valid CSV file.")

#         try:
#             df = pd.read_csv(file)
#             original_df = df.copy()
#             drift_message = ""
#             is_drift = None  # ✅ Initialize to avoid UnboundLocalError

#             # 🔍 Drift Detection (before preprocessing)
#             if os.path.exists(REFERENCE_DATA_PATH):
#                 reference_df = pd.read_csv(REFERENCE_DATA_PATH)
#                 is_drift = check_data_drift(reference_df, df)
#                 if is_drift:
#                     drift_message = "⚠️ Drift Detected!"
#                 else:
#                     drift_message = "No drift detected"
#             else:
#                 print("⚠️ Reference data not found, skipping drift detection")

#             # ⚙️ Preprocess & Predict
#             X_processed = preprocess_new_data(df)
#             model = load_model()
#             predictions = model.predict(X_processed)
#             original_df["Prediction"] = predictions

#             print(f"✅ Drift detected: {is_drift}")
#             print(f"📦 Drift message: {drift_message}")
#             return render_template(
#                 "results.html",
#                 tables=[original_df.to_html(classes='table table-bordered table-striped', index=False)],
#                 titles=original_df.columns.values,
#                 drift_message=drift_message
#             )
            

#         except Exception as e:
#             return render_template("index.html", error=f"❌ Prediction failed: {str(e)}")

#     return render_template("index.html")


# if __name__ == "__main__":
#     app.run(debug=True, port=5001)



import os
import pandas as pd
import joblib
import uuid
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
import boto3
import requests

from src.detect_drift import check_data_drift
from src.preprocessing import preprocess_data

# --- Flask app instance ---
app = Flask(__name__, template_folder="templates")

# --- Constants ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "reference_data.csv")
MODEL_URI_PATH = os.path.join(BASE_DIR, "latest_model_uri.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --- Load latest model ---
def load_model():
    if not os.path.exists(MODEL_URI_PATH):
        raise FileNotFoundError("Model URI file not found.")
    
    with open(MODEL_URI_PATH, "r") as f:
        model_uri = f.read().strip()

    model_filename = model_uri.split("/")[-1] + ".pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return joblib.load(model_path)

# --- Flask Routes ---
@app.route("/")
def index():
    return redirect(url_for("predict"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a valid CSV file.")

        try:
            df = pd.read_csv(file)
            original_df = df.copy()
            drift_message = ""
            is_drift = None

            # 🔄 Auto-generate reference data on first upload
            if not os.path.exists(REFERENCE_DATA_PATH):
                df.to_csv(REFERENCE_DATA_PATH, index=False)
            else:
                # 🔍 Drift Detection
                reference_df = pd.read_csv(REFERENCE_DATA_PATH)
                is_drift = check_data_drift(reference_df, df)
                drift_message = "⚠️ Drift Detected!" if is_drift else "✅ No Drift"
                print("msg", drift_message)

            # ⚙️ Preprocess + Predict
            X_processed, _ = preprocess_data(df, training=False)
            model = load_model()
            predictions = model.predict(X_processed)

            original_df["prediction"] = predictions
            original_df["upload_time"] = datetime.now()
            original_df["drift_flag"] = bool(is_drift)

            return render_template(
                "results.html",
                tables=[original_df.to_html(classes="table table-bordered", index=False)],
                titles=original_df.columns.values,
                drift_message=drift_message
            )

        except Exception as e:
            return render_template("index.html", error=f"❌ Prediction failed: {str(e)}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
