from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os
from src.detect_drift import check_data_drift

# Create a Flask web application instance
app = Flask(__name__, template_folder="templates")

# Define base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the file that stores the latest model URI
MODEL_URI_PATH = os.path.join(BASE_DIR, "latest_model_uri.txt")

# Directory where model files are stored
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Path to the reference data used for drift detection
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "reference_data.csv")


# Function to load the trained model based on URI stored in the text file
def load_model():
    # Raise error if model URI file is not found
    if not os.path.exists(MODEL_URI_PATH):
        raise FileNotFoundError(f"latest_model_uri.txt not found at {MODEL_URI_PATH}")
    
    # Read the model URI from the file
    with open(MODEL_URI_PATH, "r") as f:
        model_uri = f.read().strip()
    
    # Extract the model filename from the URI and form the full path
    model_filename = model_uri.split("/")[-1] + ".pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # Raise error if the model file is missing
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load and return the model
    return joblib.load(model_path)


# Function to preprocess uploaded CSV data using the custom preprocessing function
def preprocess_new_data(df):
    from src.preprocessing import preprocess_data
    # Preprocess the input data without treating it as training data
    X_processed, _ = preprocess_data(df, training=False)
    return X_processed


# Home route that redirects to the prediction page
@app.route("/")
def index():
    return redirect(url_for("predict"))


# Route to handle file upload and prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Handle POST request when a file is uploaded
    if request.method == "POST":
        # Get the uploaded file from the form
        file = request.files.get("file")

        # If file is not uploaded or not a CSV, return an error message
        if not file or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a valid CSV file.")

        try:
            # Read the uploaded CSV file into a DataFrame
            df = pd.read_csv(file)
            # Keep a copy of original data for displaying predictions
            original_df = df.copy()

            # Initialize drift message and drift flag
            drift_message = ""
            is_drift = None

            # If reference data exists, perform drift detection
            if os.path.exists(REFERENCE_DATA_PATH):
                reference_df = pd.read_csv(REFERENCE_DATA_PATH)
                is_drift = check_data_drift(reference_df, df)
                # Set message based on drift result
                if is_drift:
                    drift_message = "Drift Detected"
                else:
                    drift_message = "No drift detected"
            else:
                print("Reference data not found, skipping drift detection")

            # Preprocess the input data
            X_processed = preprocess_new_data(df)
            # Load the trained model
            model = load_model()
            # Make predictions using the model
            predictions = model.predict(X_processed)
            # Add predictions to the original DataFrame
            original_df["Prediction"] = predictions

            # Print drift and prediction info in terminal
            print(f"Drift detected: {is_drift}")
            print(f"Drift message: {drift_message}")

            # Render the results page with prediction table and drift message
            return render_template(
                "results.html",
                tables=[original_df.to_html(classes='table table-bordered table-striped', index=False)],
                titles=original_df.columns.values,
                drift_message=drift_message
            )

        # Handle any errors during the process
        except Exception as e:
            return render_template("index.html", error=f"Prediction failed: {str(e)}")

    # If method is GET, render the upload form
    return render_template("index.html")


# Run the Flask app on port 5001 in debug mode
if __name__ == "__main__":
    app.run(debug=True, port=5001)
