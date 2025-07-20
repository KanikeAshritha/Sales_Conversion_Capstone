# import pandas as pd
# import numpy as np
# import joblib
# import os

# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.base import BaseEstimator, TransformerMixin

# ARTIFACT_DIR = "artifacts"

# # --- Custom Transformers ---
# class DropColumnsTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, drop_ids=True, missing_thresh=0.4):
#         self.drop_ids = drop_ids
#         self.missing_thresh = missing_thresh
#         self.cols_to_drop_ = []

#     def fit(self, X, y=None):
#         X_ = X.copy()
#         if self.drop_ids:
#             self.cols_to_drop_ += ['Prospect ID', 'Lead Number']
#         missing_ratio = X_.isnull().mean()
#         self.cols_to_drop_ += missing_ratio[missing_ratio > self.missing_thresh].index.tolist()
#         return self

#     def transform(self, X):
#         return X.drop(columns=self.cols_to_drop_, errors='ignore')


# class ReplaceSelectTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.replace("Select", np.nan)


# class BinaryMapper(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.binary_cols = []

#     def fit(self, X, y=None):
#         self.binary_cols = [col for col in X.columns if X[col].isin(['Yes', 'No']).all()]
#         return self

#     def transform(self, X):
#         X_copy = X.copy()
#         for col in self.binary_cols:
#             X_copy[col] = X_copy[col].map({'Yes': 1, 'No': 0})
#         return X_copy

# # --- Main Preprocessing Function ---
# def preprocess_data(df, training=True):
#     df = df.copy()

#     # Ensure target exists
#     if 'Converted' not in df.columns:
#         df['Converted'] = np.nan

#     # Step 1: Drop, Replace, Map
#     if training:
#         dropper = DropColumnsTransformer()
#         selector = ReplaceSelectTransformer()
#         mapper = BinaryMapper()

#         df = dropper.fit_transform(df)
#         df = selector.transform(df)
#         df = mapper.fit_transform(df)

#         os.makedirs(ARTIFACT_DIR, exist_ok=True)
#         joblib.dump(dropper, f"{ARTIFACT_DIR}/dropper.pkl")
#         joblib.dump(mapper, f"{ARTIFACT_DIR}/mapper.pkl")
#     else:
#         dropper = joblib.load(f"{ARTIFACT_DIR}/dropper.pkl")
#         mapper = joblib.load(f"{ARTIFACT_DIR}/mapper.pkl")
#         selector = ReplaceSelectTransformer()

#         df = dropper.transform(df)
#         df = selector.transform(df)
#         df = mapper.transform(df)

#     # Step 2: Split target
#     y = df["Converted"]
#     X = df.drop(columns=["Converted"], errors="ignore")

#     # Step 3: Load/create feature definitions
#     if training:
#         num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#         cat_cols = X.select_dtypes(include='object').columns.tolist()
#         joblib.dump(num_cols, f"{ARTIFACT_DIR}/num_cols.pkl")
#         joblib.dump(cat_cols, f"{ARTIFACT_DIR}/cat_cols.pkl")
#     else:
#         num_cols = joblib.load(f"{ARTIFACT_DIR}/num_cols.pkl")
#         cat_cols = joblib.load(f"{ARTIFACT_DIR}/cat_cols.pkl")
#         for col in num_cols + cat_cols:
#             if col not in X.columns:
#                 X[col] = np.nan
#         X = X[num_cols + cat_cols]

#     # Step 4: Pipelines
#     numeric_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', MinMaxScaler())
#     ])
#     categorical_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])
#     preprocessor = ColumnTransformer([
#         ('num', numeric_pipeline, num_cols),
#         ('cat', categorical_pipeline, cat_cols)
#     ])

#     # Step 5: Fit or Load Preprocessor
#     if training:
#         X_processed = preprocessor.fit_transform(X)
#         joblib.dump(preprocessor, f"{ARTIFACT_DIR}/preprocessor.pkl")
#         feature_names = preprocessor.get_feature_names_out()
#         joblib.dump(feature_names, os.path.join(ARTIFACT_DIR, "expected_features.pkl"))
#     else:
#         preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.pkl")
#         X_processed = preprocessor.transform(X)
#         feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "expected_features.pkl"))

#         print(f"🔎 Inference: X shape before transform: {X.shape}")
#         print(f"✅ Transformed shape: {X_processed.shape}")
#         print(f"📦 Expected features: {len(feature_names)}")

#     # Step 6: Ensure consistent output DataFrame
#     X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

#     if not training:
#         expected_features = joblib.load(os.path.join(ARTIFACT_DIR, "expected_features.pkl"))

#         # Add missing expected columns
#         for col in expected_features:
#             if col not in X_processed_df.columns:
#                 X_processed_df[col] = 0

#         # Drop extra columns
#         X_processed_df = X_processed_df[expected_features]

#     return X_processed_df, y.reset_index(drop=True)
 
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Custom Transformers ---
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_ids=True, missing_thresh=0.4):
        self.drop_ids = drop_ids
        self.missing_thresh = missing_thresh
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        X_ = X.copy()
        if self.drop_ids:
            self.cols_to_drop_ += ['Prospect ID', 'Lead Number']
        missing_ratio = X_.isnull().mean()
        self.cols_to_drop_ += missing_ratio[missing_ratio > self.missing_thresh].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors='ignore')

class ReplaceSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.select_cols = None

    def fit(self, X, y=None):
        self.select_cols = [col for col in X.columns if X[col].astype(str).str.contains("Select", case=False, na=False).any()]
        return self

    def transform(self, X):
        for col in self.select_cols:
            X[col] = X[col].replace("Select", np.nan)
        return X

class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_map = {"Yes": 1, "No": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.binary_map)

# --- Preprocessing Pipeline ---
def preprocess_data(df: pd.DataFrame, training: bool = True):
    label_column = "Converted"
    
    df = df.copy()
    
    # Drop columns transformer
    dropper = DropColumnsTransformer()
    df = dropper.transform(df)
    
    # Binary yes/no map
    bin_mapper = BinaryMapper()
    df = bin_mapper.fit_transform(df)
    
    # Replace "Select" entries
    select_replacer = ReplaceSelectTransformer()
    df = select_replacer.fit_transform(df)
    
    if training:
        y = df[label_column].squeeze()
        X = df.drop(columns=[label_column])
    else:
        if label_column in df.columns:
            y = df[label_column].squeeze()
            X = df.drop(columns=[label_column])
        else:
            y = None
            X = df.copy()
    
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    
    # Save columns
    if training:
        joblib.dump(num_cols, os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
        joblib.dump(cat_cols, os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))
    else:
        num_cols = joblib.load(os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
        cat_cols = joblib.load(os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))

    # Ensure all expected columns are present (fill missing as NaN)
    for col in num_cols + cat_cols:
        if col not in X.columns:
            X[col] = np.nan

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    if training:
        X_processed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        joblib.dump(dropper, os.path.join(ARTIFACT_DIR, "dropper.pkl"))
        joblib.dump(bin_mapper, os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
        joblib.dump(select_replacer, os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(ARTIFACT_DIR, "input_columns.pkl"))
        joblib.dump(preprocessor.get_feature_names_out(), os.path.join(ARTIFACT_DIR, "expected_features.pkl"))
    else:
        preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        dropper = joblib.load(os.path.join(ARTIFACT_DIR, "dropper.pkl"))
        bin_mapper = joblib.load(os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
        select_replacer = joblib.load(os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
        expected_features = joblib.load(os.path.join(ARTIFACT_DIR, "expected_features.pkl"))
        
        X_processed = preprocessor.transform(X)
        X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
        
        # Align to expected features
        for col in expected_features:
            if col not in X_processed_df.columns:
                X_processed_df[col] = 0
        for col in X_processed_df.columns:
            if col not in expected_features:
                X_processed_df.drop(columns=[col], inplace=True)
        X_processed_df = X_processed_df[expected_features]

        return X_processed_df, y
    
    return pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out()), y
