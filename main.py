# import pandas as pd
# from src.postgres_utils import load_data_from_postgres
# from src.preprocessing import preprocess_data
# from src.mlflow_runner import mlflow_run_with_grid_search  
# from src.model_training import evaluate_model

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import train_test_split
# import os
# import warnings
# import logging

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# logging.getLogger("lightgbm").setLevel(logging.ERROR)
# logging.getLogger("mlflow").setLevel(logging.ERROR)

# DB_HOST = os.getenv("DB_HOST", "localhost")
# db_config = {
#     "dbname": "sales_conversion",
#     "user": "kanikeashritha",
#     "password": "ash",
#     "host": DB_HOST,
#     "port": "5432"
# }

# def main():
#     # from src.shap import explain_model_with_shap

#     print("🗄️ Loading data from PostgreSQL...")
#     df = pd.read_csv("data/raw/Lead Scoring.csv")
#     df.to_csv("airflow/data/new_data.csv", index=False) 

#     def clean_column_names(columns):
#         return columns.str.replace(r'[{}[\]<>"\',: ]', '_', regex=True)

#     # ✅ Now perform full preprocessing (encoding, imputation etc.)
#     print("⚙️ Running full preprocessing...")
#     X_processed_df,y= preprocess_data(df, training=True)
#     # Save cleaned feature structure

    
# # # After preprocessing
#     X_processed_df.columns = clean_column_names(X_processed_df.columns)
#     df_cleaned = pd.concat([X_processed_df, y], axis=1)

#     print("✂️ Splitting data...")
#     X = df_cleaned.drop(columns=["Converted"])
#     y = df_cleaned["Converted"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     # ✅ Model registry for classification with param grids
#     model_registry = {
#         "LogisticRegression": (
#             LogisticRegression(class_weight='balanced'),
#             {"model__C": [0.1, 1.0, 10.0]}
#         ),
#         "RandomForest": (
#             RandomForestClassifier(class_weight='balanced'),
#             {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]}
#         ),
#         "XGBoost": (
#             XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
#             {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}
#         ),
#         "LightGBM": (
#             LGBMClassifier(verbose=-1),
#             {"model__n_estimators": [100, 200], "model__num_leaves": [31, 50]}
#         ),
#         "DecisionTree": (
#             DecisionTreeClassifier(),
#             {"model__max_depth": [5, 10, None]}
#         )
#     }

#     for model_name, (model, _) in model_registry.items():
#         evaluate_model(X_train, y_train, X_test, y_test, model_name, model)


#     # mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry)

#     # Optional SHAP explainability:
#     # sample_X = df_cleaned.sample(100, random_state=42).drop(columns=["Converted"])
#     # explain_model_with_shap(sample_X)

# if __name__ == "__main__":
#     main()



import pandas as pd
from src.postgres_utils import load_data_from_postgres
from src.preprocessing import preprocess_data
from src.mlflow_runner import mlflow_run_with_grid_search  
from src.model_training import evaluate_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)

DB_HOST = os.getenv("DB_HOST", "localhost")
db_config = {
    "dbname": "sales_conversion",
    "user": "kanikeashritha",
    "password": "ash",
    "host": DB_HOST,
    "port": "5432"
}

def main():
    # from src.shap import explain_model_with_shap

    print("🗄️ Loading data from postgres...")
    df = load_data_from_postgres()

    def clean_column_names(columns):
        return columns.str.replace(r'[{}[\]<>"\',: ]', '_', regex=True)

    # ✅ Now perform full preprocessing (encoding, imputation etc.)
    print("⚙️ Running full preprocessing...")
    X_processed_df, y= preprocess_data(df, training=True)
    
# After preprocessing
    X_processed_df.columns = clean_column_names(X_processed_df.columns)
    df_cleaned = pd.concat([X_processed_df, y], axis=1)

    print("✂️ Splitting data...")
    X = df_cleaned.drop(columns=["Converted"])
    y = df_cleaned["Converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ✅ Model registry for classification with param grids
    model_registry = {
        "LogisticRegression": (
            LogisticRegression(class_weight='balanced'),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight='balanced'),
            {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}
        ),
        "LightGBM": (
            LGBMClassifier(verbose=-1),
            {"model__n_estimators": [100, 200], "model__num_leaves": [31, 50]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(),
            {"model__max_depth": [5, 10, None]}
        )
    }

    for model_name, (model, _) in model_registry.items():
        evaluate_model(X_train, y_train, X_test, y_test, model_name, model)


    mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry)

    # Optional SHAP explainability:
    # sample_X = df_cleaned.sample(100, random_state=42).drop(columns=["Converted"])
    # explain_model_with_shap(sample_X)

if __name__ == "__main__":
    main()