import pandas as pd
import numpy as np
import os
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Creates directory to store saved transformers and column names
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Custom Transformers ---

# Drops unwanted columns from the dataframe
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# Replaces values like "Select" with NaN in specific columns
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

# Converts binary yes/no values to 1/0
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_map = {"Yes": 1, "No": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.binary_map)

# --- Preprocessing Pipeline ---

def preprocess_data(df: pd.DataFrame, training: bool = True):
    # List of columns to remove from the dataset
    drop_columns = [
        "Prospect ID", "Lead Number", "Tags", "Last Activity", 
        "What matters most to you in choosing a course", "Last Notable Activity"
    ]
    label_column = "Converted"
    
    df = df.copy()
    
    # Removes specified columns using DropColumnsTransformer
    dropper = DropColumnsTransformer(columns_to_drop=drop_columns)
    df = dropper.fit_transform(df)
    
    # Replaces yes/no text with 1/0 using BinaryMapper
    bin_mapper = BinaryMapper()
    df = bin_mapper.fit_transform(df)
    
    # Replaces "Select" text with NaN using ReplaceSelectTransformer
    select_replacer = ReplaceSelectTransformer()
    df = select_replacer.fit_transform(df)
    
    # Splits features and label based on whether it is training or prediction
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
    
    # Identifies numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    
    # Saves identified column names to disk if training
    if training:
        joblib.dump(num_cols, os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
        joblib.dump(cat_cols, os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))
    else:
        # Loads previously saved column names during prediction
        num_cols = joblib.load(os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
        cat_cols = joblib.load(os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))

    # Ensures all expected columns are present in the input data
    for col in num_cols + cat_cols:
        if col not in X.columns:
            X[col] = np.nan

    # Pipeline for handling numeric columns: fill missing with mean and scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    # Pipeline for categorical columns: fill missing with mode and encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combines numeric and categorical pipelines into a single transformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ], verbose_feature_names_out=False)

    if training:
        # Fits and transforms training data using the complete pipeline
        X_processed = preprocessor.fit_transform(X)

        # Saves all components to disk for use during inference
        joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        joblib.dump(dropper, os.path.join(ARTIFACT_DIR, "dropper.pkl"))
        joblib.dump(bin_mapper, os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
        joblib.dump(select_replacer, os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(ARTIFACT_DIR, "input_columns.pkl"))
        joblib.dump(preprocessor.get_feature_names_out(), os.path.join(ARTIFACT_DIR, "expected_features.pkl"))
    else:
        # Loads saved transformers and column names
        preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        dropper = joblib.load(os.path.join(ARTIFACT_DIR, "dropper.pkl"))
        bin_mapper = joblib.load(os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
        select_replacer = joblib.load(os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
        expected_features = joblib.load(os.path.join(ARTIFACT_DIR, "expected_features.pkl"))
        
        # Transforms data using already fitted transformer
        X_processed = preprocessor.transform(X)
        X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
        
        # Ensures output DataFrame has exact expected features
        for col in expected_features:
            if col not in X_processed_df.columns:
                X_processed_df[col] = 0
        for col in X_processed_df.columns:
            if col not in expected_features:
                X_processed_df.drop(columns=[col], inplace=True)
        X_processed_df = X_processed_df[expected_features]

        return X_processed_df, y

    # Returns processed training data as DataFrame
    return pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out()), y


# import pandas as pd
# import numpy as np
# import os
# import joblib

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# from sklearn.model_selection import StratifiedKFold

# from imblearn.over_sampling import SMOTE
# from scipy.stats.mstats import winsorize

# # Create artifacts directory
# ARTIFACT_DIR = "artifacts"
# os.makedirs(ARTIFACT_DIR, exist_ok=True)

# # --- Custom Transformers ---

# class DropColumnsTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns_to_drop=None):
#         self.columns_to_drop = columns_to_drop

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.drop(columns=self.columns_to_drop, errors='ignore')

# class ReplaceSelectTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.select_cols = None

#     def fit(self, X, y=None):
#         self.select_cols = [col for col in X.columns if X[col].astype(str).str.contains("Select", case=False, na=False).any()]
#         return self

#     def transform(self, X):
#         for col in self.select_cols:
#             X[col] = X[col].replace("Select", np.nan)
#         return X

# class BinaryMapper(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.binary_map = {"Yes": 1, "No": 0}

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.replace(self.binary_map)

# class Winsorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, limits=(0.01, 0.01)):  # Winsorize 1% on each side
#         self.limits = limits
#         self.num_cols = []

#     def fit(self, X, y=None):
#         self.num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#         return self

#     def transform(self, X):
#         X_copy = X.copy()
#         for col in self.num_cols:
#             X_copy[col] = winsorize(X_copy[col], limits=self.limits)
#         return X_copy

# # --- Preprocessing Pipeline Function ---

# def preprocess_data(df: pd.DataFrame, training: bool = True, apply_smote: bool = False):
#     df = df.copy()
#     drop_columns = [
#         "Prospect ID", "Lead Number", "Tags", "Last Activity", 
#         "What matters most to you in choosing a course", "Last Notable Activity"
#     ]
#     label_column = "Converted"


#     # Apply custom transformers
#     dropper = DropColumnsTransformer(columns_to_drop=drop_columns)
#     df = dropper.fit_transform(df)

#     bin_mapper = BinaryMapper()
#     df = bin_mapper.fit_transform(df)

#     select_replacer = ReplaceSelectTransformer()
#     df = select_replacer.fit_transform(df)

#     winsorizer = Winsorizer()
#     df = winsorizer.fit_transform(df)

#     # Split label and features
#     if training:
#         y = df[label_column].squeeze()
#         X = df.drop(columns=[label_column])
#     else:

#         if label_column in df.columns:
#             y = df[label_column].squeeze()
#             X = df.drop(columns=[label_column])
#         else:
#             y = None
#             X = df.copy()

#     num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

#     if training:
#         joblib.dump(num_cols, os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
#         joblib.dump(cat_cols, os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))
#     else:
#         num_cols = joblib.load(os.path.join(ARTIFACT_DIR, "num_cols.pkl"))
#         cat_cols = joblib.load(os.path.join(ARTIFACT_DIR, "cat_cols.pkl"))

#     for col in num_cols + cat_cols:
#         if col not in X.columns:
#             X[col] = np.nan

#     numeric_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="mean")),
#         ("scaler", MinMaxScaler())
#     ])

#     categorical_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#     ])

#     preprocessor = ColumnTransformer([
#         ("num", numeric_pipeline, num_cols),
#         ("cat", categorical_pipeline, cat_cols)
#     ], verbose_feature_names_out=False)

#     if training:
#         X_processed = preprocessor.fit_transform(X)

#         joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
#         joblib.dump(dropper, os.path.join(ARTIFACT_DIR, "dropper.pkl"))
#         joblib.dump(bin_mapper, os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
#         joblib.dump(select_replacer, os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
#         joblib.dump(winsorizer, os.path.join(ARTIFACT_DIR, "winsorizer.pkl"))
#         joblib.dump(X.columns.tolist(), os.path.join(ARTIFACT_DIR, "input_columns.pkl"))
#         joblib.dump(preprocessor.get_feature_names_out(), os.path.join(ARTIFACT_DIR, "expected_features.pkl"))

#         X_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

#         # 🧪 Optional: Apply SMOTE to balance classes
#         if apply_smote:
#             smote = SMOTE(random_state=42)
#             X_df, y = smote.fit_resample(X_df, y)

#         return X_df,y

#     else:
#         preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
#         dropper = joblib.load(os.path.join(ARTIFACT_DIR, "dropper.pkl"))
#         bin_mapper = joblib.load(os.path.join(ARTIFACT_DIR, "bin_mapper.pkl"))
#         select_replacer = joblib.load(os.path.join(ARTIFACT_DIR, "select_replacer.pkl"))
#         winsorizer = joblib.load(os.path.join(ARTIFACT_DIR, "winsorizer.pkl"))
#         expected_features = joblib.load(os.path.join(ARTIFACT_DIR, "expected_features.pkl"))

#         X = dropper.transform(X)
#         X = bin_mapper.transform(X)
#         X = select_replacer.transform(X)
#         X = winsorizer.transform(X)

#         X_processed = preprocessor.transform(X)
#         X_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

#         for col in expected_features:
#             if col not in X_df.columns:
#                 X_df[col] = 0
#         for col in X_df.columns:
#             if col not in expected_features:
#                 X_df.drop(columns=[col], inplace=True)
#         X_df = X_df[expected_features]
#         return X_df, y

#     return pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out()), y

# # --- Stratified K-Fold Setup (for training loop) ---

# def get_stratified_kfold(n_splits=5, random_state=42):
#     return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
