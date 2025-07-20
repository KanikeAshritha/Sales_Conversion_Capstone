import shap
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd

def explain_model_with_shap(X_raw):
    # Displays message to indicate that SHAP explainability is running
    print("Running SHAP explainability...")

    # Defines the path to the saved model file
    model_path = os.path.join("models", "model.pkl")

    # Checks whether the model file exists
    if not os.path.exists(model_path):
        print("model.pkl not found.")
        return

    # Loads the trained pipeline which contains preprocessing and model steps
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    # Preprocesses the input data using the same logic used during model training
    from src.preprocessing import preprocess_data
    X_processed_df, _ = preprocess_data(X_raw, training=False)

    # Creates a SHAP explainer using the model's predict function and the preprocessed data
    explainer = shap.Explainer(model.predict, X_processed_df)
    shap_values = explainer(X_processed_df)

    # Creates the directory to store SHAP plots if it does not exist
    os.makedirs("artifacts/shap", exist_ok=True)

    # Generates and saves SHAP bar plot that shows average importance of each feature
    shap.plots.bar(shap_values, show=False)
    plt.savefig("artifacts/shap/shap_summary_bar.png")
    plt.clf()

    # Generates and saves SHAP beeswarm plot that shows feature effects for all samples
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("artifacts/shap/shap_beeswarm.png")
    plt.clf()

    # Generates and saves SHAP waterfall plot that shows contribution of features for one sample
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("artifacts/shap/shap_waterfall_sample0.png")
    plt.clf()

    # Calculates and displays top 5 features based on mean absolute SHAP values
    print("Top SHAP Features:")
    import numpy as np
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    importance_df = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    print(importance_df.head(5))

    # Displays confirmation that plots have been saved
    print("SHAP plots saved in artifacts/shap/")
