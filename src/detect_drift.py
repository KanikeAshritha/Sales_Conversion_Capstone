# import os
# from evidently import Report
# from evidently.presets import DataDriftPreset
# import pandas as pd
# import numpy as np

# DRIFT_THRESHOLD = 0.3
# REFERENCE_PATH = "artifacts/reference_data.csv"
# CURRENT_PATH = "data/new_data.csv"
# DRIFT_FLAG_PATH = "/opt/airflow/dags/drift_flag.json"


# def check_data_drift(reference_df, current_df, save_dir="artifacts"):
#     reference_df = reference_df.drop(columns=["Converted"], errors="ignore")
#     current_df = current_df.drop(columns=["Converted"], errors="ignore")

#     # Align dtypes
#     for col in reference_df.columns:
#         if col in current_df.columns:
#             ref_dtype = reference_df[col].dtype
#             try:
#                 if pd.api.types.is_numeric_dtype(ref_dtype):
#                     current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
#                 elif pd.api.types.is_bool_dtype(ref_dtype):
#                     current_df[col] = current_df[col].astype(bool)
#             except Exception as e:
#                 print(f"⚠️ Failed to cast column {col}: {e}")
#         else:
#             current_df[col] = np.nan

#     current_df = current_df[reference_df.columns]

#     def drop_empty_or_constant(df1, df2):
#         drop_cols = []
#         for col in df1.columns:
#             if df1[col].dropna().nunique() <= 1 or df2[col].dropna().nunique() <= 1:
#                 drop_cols.append(col)
#         return df1.drop(columns=drop_cols), df2.drop(columns=drop_cols)

#     reference_df, current_df = drop_empty_or_constant(reference_df, current_df)

#     if reference_df.shape[1] == 0 or current_df.shape[1] == 0:
#         print("⚠️ Drift check skipped: No usable columns after filtering.")
#         return False

#     report = Report(metrics=[DataDriftPreset()])
#     res = report.run(reference_data=reference_df, current_data=current_df)

#     os.makedirs(save_dir, exist_ok=True)
#     res.save_html(os.path.join(save_dir, "drift_report.html"))
#     res.save_json(os.path.join(save_dir, "drift_report.json"))

#     drift_results = report.as_dict()
#     try:
#         n_drifted = drift_results["metrics"][0]["result"]["number_of_drifted_columns"]
#         total = drift_results["metrics"][0]["result"]["number_of_columns"]
#         drift_score = n_drifted / total
#         drift_detected = drift_score > DRIFT_THRESHOLD
#         print(f"[✅ Drift Check] Drift Detected: {drift_detected} ({n_drifted}/{total})")
#         return drift_detected
#     except Exception as e:
#         print(f"⚠️ Failed to parse drift result: {e}")
#         return False

# def run_drift_detection():
#     if not os.path.exists(REFERENCE_PATH) or not os.path.exists(CURRENT_PATH):
#         print("Missing reference or current data file.")
#         return

#     ref_df = pd.read_csv(REFERENCE_PATH)
#     cur_df = pd.read_csv(CURRENT_PATH)
#     drift = check_data_drift(ref_df, cur_df)

#     with open(DRIFT_FLAG_PATH, "w") as f:
#         json.dump({"drift_detected": drift}, f)


#     print(f"[📌 Drift Flag Saved] {drift}")
#     print("[✅ Drift detection complete]")
#     print(f"Drift flag written to: {DRIFT_FLAG_PATH}")


# import os
# from evidently import Report
# from evidently.presets import DataDriftPreset
# import pandas as pd
# import numpy as np

# DRIFT_THRESHOLD = 0.3
# REFERENCE_PATH = "artifacts/reference_data.csv"
# CURRENT_PATH = "data/new_data.csv"
# DRIFT_FLAG_PATH = "artifacts/drift_flag.txt"

# def check_data_drift(reference_df, current_df, save_dir="artifacts"):
#     reference_df = reference_df.drop(columns=["Converted"], errors="ignore")
#     current_df = current_df.drop(columns=["Converted"], errors="ignore")

#     # Align dtypes
#     for col in reference_df.columns:
#         if col in current_df.columns:
#             ref_dtype = reference_df[col].dtype
#             try:
#                 if pd.api.types.is_numeric_dtype(ref_dtype):
#                     current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
#                 elif pd.api.types.is_bool_dtype(ref_dtype):
#                     current_df[col] = current_df[col].astype(bool)
#             except Exception as e:
#                 print(f"⚠️ Failed to cast column {col}: {e}")
#         else:
#             current_df[col] = np.nan

#     current_df = current_df[reference_df.columns]

#     def drop_empty_or_constant(df1, df2):
#         drop_cols = []
#         for col in df1.columns:
#             if df1[col].dropna().nunique() <= 1 or df2[col].dropna().nunique() <= 1:
#                 drop_cols.append(col)
#         return df1.drop(columns=drop_cols), df2.drop(columns=drop_cols)

#     reference_df, current_df = drop_empty_or_constant(reference_df, current_df)

#     if reference_df.shape[1] == 0 or current_df.shape[1] == 0:
#         print("⚠️ Drift check skipped: No usable columns after filtering.")
#         return False

#     report = Report(metrics=[DataDriftPreset()])
#     res = report.run(reference_data=reference_df, current_data=current_df)

#     os.makedirs(save_dir, exist_ok=True)
#     res.save_html(os.path.join(save_dir, "drift_report.html"))
#     res.save_json(os.path.join(save_dir, "drift_report.json"))

#     drift_results = report.as_dict()
#     try:
#         n_drifted = drift_results["metrics"][0]["result"]["number_of_drifted_columns"]
#         total = drift_results["metrics"][0]["result"]["number_of_columns"]
#         drift_score = n_drifted / total
#         drift_detected = drift_score > DRIFT_THRESHOLD
#         print(f"[✅ Drift Check] Drift Detected: {drift_detected} ({n_drifted}/{total})")
#         return drift_detected
#     except Exception as e:
#         print(f"⚠️ Failed to parse drift result: {e}")
#         return False

# def run_drift_detection():
#     if not os.path.exists(REFERENCE_PATH) or not os.path.exists(CURRENT_PATH):
#         print("Missing reference or current data file.")
#         return

#     ref_df = pd.read_csv(REFERENCE_PATH)
#     cur_df = pd.read_csv(CURRENT_PATH)
#     drift = check_data_drift(ref_df, cur_df)

#     with open(DRIFT_FLAG_PATH, "w") as f:
#         json.dump({"drift_detected": drift}, f)

#     print(f"[📌 Drift Flag Saved] {drift}")





# ✅ src/detect_drift.py
import os
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import numpy as np
import json

DRIFT_THRESHOLD = 0.3
REFERENCE_PATH = "artifacts/reference_data.csv"
CURRENT_PATH = "data/new_data.csv"
DRIFT_FLAG_PATH = "artifacts/drift_flag.txt"

def check_data_drift(reference_df, current_df, save_dir="artifacts"):
    reference_df = reference_df.drop(columns=["Converted"], errors="ignore")
    current_df = current_df.drop(columns=["Converted"], errors="ignore")

    # Align dtypes
    for col in reference_df.columns:
        if col in current_df.columns:
            ref_dtype = reference_df[col].dtype
            try:
                if pd.api.types.is_numeric_dtype(ref_dtype):
                    current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
                elif pd.api.types.is_bool_dtype(ref_dtype):
                    current_df[col] = current_df[col].astype(bool)
            except Exception as e:
                print(f"⚠️ Failed to cast column {col}: {e}")
        else:
            current_df[col] = np.nan

    current_df = current_df[reference_df.columns]

    def drop_empty_or_constant(df1, df2):
        drop_cols = []
        for col in df1.columns:
            if df1[col].dropna().nunique() <= 1 or df2[col].dropna().nunique() <= 1:
                drop_cols.append(col)
        return df1.drop(columns=drop_cols), df2.drop(columns=drop_cols)

    reference_df, current_df = drop_empty_or_constant(reference_df, current_df)

    if reference_df.shape[1] == 0 or current_df.shape[1] == 0:
        print("⚠️ Drift check skipped: No usable columns after filtering.")
        return False

    report = Report(metrics=[DataDriftPreset()])
    res = report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(save_dir, exist_ok=True)
    res.save_html(os.path.join(save_dir, "drift_report.html"))
    res.save_json(os.path.join(save_dir, "drift_report.json"))

    with open(os.path.join(save_dir, "drift_report.json"), "r") as f:
        drift_results = json.load(f)
    try:
        n_drifted = drift_results["metrics"][0]["result"]["number_of_drifted_columns"]
        total = drift_results["metrics"][0]["result"]["number_of_columns"]
        drift_score = n_drifted / total
        drift_detected = drift_score > DRIFT_THRESHOLD
        print(f"[✅ Drift Check] Drift Detected: {drift_detected} ({n_drifted}/{total})")
        return drift_detected
    except Exception as e:
        print(f"⚠️ Failed to parse drift result: {e}")
        return False

def run_drift_detection():
    if not os.path.exists(REFERENCE_PATH) or not os.path.exists(CURRENT_PATH):
        print("Missing reference or current data file.")
        return

    ref_df = pd.read_csv(REFERENCE_PATH)
    cur_df = pd.read_csv(CURRENT_PATH)
    drift = check_data_drift(ref_df, cur_df)

    with open(DRIFT_FLAG_PATH, "w") as f:
        json.dump({"drift_detected": drift}, f)

    print(f"[📌 Drift Flag Saved] {drift}")
