# Import required libraries
import pandas as pd
import os
import warnings
import logging

# Import preprocessing, model evaluation, and MLflow tracking functions
from src.preprocessing import preprocess_data
from src.mlflow_runner import mlflow_run_with_grid_search  
from src.model_training import evaluate_model
from src.postgres_utils import load_data_from_postgres
from src.eda import exploratory_data_analysis
from src.shap import explain_model_with_shap

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

# Import train-test split
from sklearn.model_selection import train_test_split

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Set up database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
db_config = {
    "dbname": "sales_conversion",
    "user": "kanikeashritha",
    "password": "ash",
    "host": DB_HOST,
    "port": "5432"
}

# Define the main function that performs data loading, preprocessing, training, and tracking
def main():
    # Load the dataset from a CSV file
    print("Loading data...")
    df = pd.read_csv('Lead Scoring.csv')
    
    # exploratory_data_analysis(df)
    # Helper function to clean column names (remove special characters)
    def clean_column_names(columns):
        return columns.str.replace(r'[{}[\]<>"\',: ]', '_', regex=True)

    # Preprocess the data (encoding, imputation, scaling, etc.)
    print("Running full preprocessing...")
    X_processed_df, y = preprocess_data(df, training=True)

    # Clean column names to ensure compatibility with models and MLflow
    X_processed_df.columns = clean_column_names(X_processed_df.columns)

    # Recombine features and target into a clean DataFrame
    df_cleaned = pd.concat([X_processed_df, y], axis=1)

    # Split the cleaned data into features and target
    print("Splitting data...")
    X = df_cleaned.drop(columns=["Converted"])
    y = df_cleaned["Converted"]

    # Perform train-test split (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define a dictionary of models with hyperparameter grids
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

    # Evaluate each model using default settings before grid search
    for model_name, (model, _) in model_registry.items():
        evaluate_model(X_train, y_train, X_test, y_test, model_name, model)

    # Run grid search and track the best models using MLflow
    mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry)

    # Optional: run SHAP analysis for explainability (disabled by default)
    # sample_X = df_cleaned.sample(100, random_state=42).drop(columns=["Converted"])
    # explain_model_with_shap(sample_X)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()



# # Import required libraries
# import pandas as pd
# import os
# import warnings
# import logging

# # Import preprocessing, model evaluation, and MLflow tracking functions
# from src.preprocessing import preprocess_data
# from src.mlflow_runner import mlflow_run_with_grid_search
# from src.model_training import evaluate_model

# # Import machine learning models
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Import train-test split utility
# from sklearn.model_selection import train_test_split

# # Suppress unnecessary warnings for cleaner output
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# logging.getLogger("lightgbm").setLevel(logging.ERROR)
# logging.getLogger("mlflow").setLevel(logging.ERROR)

# # Set up database connection configuration (if needed for PostgreSQL)
# DB_HOST = os.getenv("DB_HOST", "localhost")
# db_config = {
#     "dbname": "sales_conversion",
#     "user": "kanikeashritha",
#     "password": "ash",
#     "host": DB_HOST,
#     "port": "5432"
# }

# # Define the main function for executing the ML pipeline
# def main():
#     # Load dataset from CSV file
#     print("Loading data...")
#     df = pd.read_csv("Lead Scoring.csv")

#     # Clean column names to ensure compatibility with ML models and MLflow
#     def clean_column_names(columns):
#         return columns.str.replace(r'[{}[\]<>"\',: ]', '_', regex=True)

#     # Run data preprocessing: encoding, imputation, scaling, etc.
#     print("Running preprocessing...")
#     X_processed_df, y = preprocess_data(df, training=True)
#     X_processed_df.columns = clean_column_names(X_processed_df.columns)

#     # Combine processed features with target for a clean dataset
#     df_cleaned = pd.concat([X_processed_df, y], axis=1)

#     # Split cleaned data into features (X) and target (y)
#     print("Splitting data...")
#     X = df_cleaned.drop(columns=["Converted"])
#     y = df_cleaned["Converted"]

#     # Perform train-test split with stratification to maintain class balance
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     # Define models and their corresponding hyperparameter grids for GridSearchCV
#     model_registry = {
#         # Logistic Regression with regularization parameter tuning
#         "LogisticRegression": (
#             LogisticRegression(class_weight='balanced', solver='liblinear'),
#             {
#                 "model__C": [0.01, 0.1, 1.0, 10.0],  # Regularization strength
#                 "model__penalty": ["l1", "l2"]       # Regularization type
#             }
#         ),
#         # Random Forest with estimators and depth tuning
#         "RandomForest": (
#             RandomForestClassifier(class_weight='balanced', random_state=42),
#             {
#                 "model__n_estimators": [100, 200, 300],  # Number of trees
#                 "model__max_depth": [5, 10, None],       # Tree depth
#                 "model__min_samples_split": [2, 5],      # Node split control
#                 "model__max_features": ["sqrt", "log2"]  # Feature subset size
#             }
#         ),
#         # XGBoost with learning rate and depth tuning
#         "XGBoost": (
#             XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#             {
#                 "model__n_estimators": [100, 150, 200],    # Boosting rounds
#                 "model__learning_rate": [0.01, 0.05, 0.1], # Learning rate
#                 "model__max_depth": [3, 5, 7],             # Max depth of trees
#                 "model__subsample": [0.8, 1.0]             # Row sampling
#             }
#         ),
#         # LightGBM with boosting and tree complexity tuning
#         "LightGBM": (
#             LGBMClassifier(verbose=-1, random_state=42),
#             {
#                 "model__n_estimators": [100, 200, 300],  # Number of boosting rounds
#                 "model__learning_rate": [0.01, 0.05],    # Learning rate
#                 "model__num_leaves": [31, 50, 70],       # Leaf size
#                 "model__max_depth": [5, 10, -1]          # Max tree depth
#             }
#         ),
#         # Decision Tree with depth and splitting strategy tuning
#         "DecisionTree": (
#             DecisionTreeClassifier(random_state=42),
#             {
#                 "model__max_depth": [5, 10, None],            # Maximum depth of tree
#                 "model__min_samples_split": [2, 5, 10],       # Minimum number of samples required to split
#                 "model__criterion": ["gini", "entropy"]       # Splitting criterion
#             }
#         )
#     }

#     # Evaluate baseline performance of all models before tuning
#     for model_name, (model, _) in model_registry.items():
#         print(f"\nTraining model: {model_name}")
#         evaluate_model(X_train, y_train, X_test, y_test, model_name, model)

#     # Run GridSearchCV and track best model for each using MLflow
#     mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry)

# if __name__ == "__main__":
#     main()
