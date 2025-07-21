import os
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import numpy as np
import json

# Threshold to determine if data drift is significant
DRIFT_THRESHOLD = 0.3

# File paths for reference and current datasets
REFERENCE_PATH = "artifacts/reference_data.csv"
CURRENT_PATH = "data/new_data.csv"
DRIFT_FLAG_PATH = "artifacts/drift_flag.txt"

# Function to check data drift between reference and current data
def check_data_drift(reference_df, current_df, save_dir="artifacts"):
    # Drop label column if it exists
    reference_df = reference_df.drop(columns=["Converted"], errors="ignore")
    current_df = current_df.drop(columns=["Converted"], errors="ignore")

    # Align datatypes between reference and current datasets
    for col in reference_df.columns:
        if col in current_df.columns:
            ref_dtype = reference_df[col].dtype
            try:
                if pd.api.types.is_numeric_dtype(ref_dtype):
                    current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
                elif pd.api.types.is_bool_dtype(ref_dtype):
                    current_df[col] = current_df[col].astype(bool)
            except Exception as e:
                print(f"Failed to cast column {col}: {e}")
        else:
            current_df[col] = np.nan

    # Reorder columns in current data to match reference
    current_df = current_df[reference_df.columns]

    # Remove columns with only one unique value or all missing in either dataset
    def drop_empty_or_constant(df1, df2):
        drop_cols = []
        for col in df1.columns:
            if df1[col].dropna().nunique() <= 1 or df2[col].dropna().nunique() <= 1:
                drop_cols.append(col)
        return df1.drop(columns=drop_cols), df2.drop(columns=drop_cols)

    reference_df, current_df = drop_empty_or_constant(reference_df, current_df)

    # If no usable columns are left, skip drift check
    if reference_df.shape[1] == 0 or current_df.shape[1] == 0:
        print("Drift check skipped: No usable columns after filtering.")
        return False

    # Generate Evidently report for data drift
    report = Report(metrics=[DataDriftPreset()])
    res = report.run(reference_data=reference_df, current_data=current_df)

    # Save report as HTML and JSON
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "drift_report.json")
    res.save_html(os.path.join(save_dir, "drift_report.html"))
    res.save_json(json_path)

    # Read drift results from JSON and evaluate drift condition
    try:
        with open(json_path, "r") as f:
            drift_results = json.load(f)

        for metric in drift_results.get("metrics", []):
            result = metric.get("result", {})
            n_drifted = result.get("number_of_drifted_columns", 0)
            total = result.get("number_of_columns", 1)
            drift_score = n_drifted / total
            drift_detected = drift_score > DRIFT_THRESHOLD
            print(f"Drift Detected: {drift_detected}")
            return drift_detected

        print("Drift metric not found in the report.")
        return False
    except Exception as e:
        print(f"Failed to parse drift result: {e}")
        return False

# Function to run the drift check and save the result to a flag file
def run_drift_detection():
    # Check if required input files exist
    if not os.path.exists(REFERENCE_PATH) or not os.path.exists(CURRENT_PATH):
        print("Missing reference or current data file.")
        return

    # Load reference and current datasets
    ref_df = pd.read_csv(REFERENCE_PATH)
    cur_df = pd.read_csv(CURRENT_PATH)

    # Perform drift check
    drift = check_data_drift(ref_df, cur_df)

    # Save the drift result to a flag file
    with open(DRIFT_FLAG_PATH, "w") as f:
        f.write("True" if drift else "False")

    print(f"[Drift Flag Saved] {drift}")
