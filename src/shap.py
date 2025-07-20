import shap
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd

def explain_model_with_shap(X_raw):
    print("🔎 Running SHAP explainability...")

    # Load model pipeline
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        print("❌ model.pkl not found.")
        return

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    # Use the same preprocessing logic to transform and align
    from src.preprocessing import preprocess_data
    X_processed_df, _ = preprocess_data(X_raw, training=False)  # returns DataFrame

    # SHAP expects a DataFrame with named columns
    explainer = shap.Explainer(model.predict, X_processed_df)  # Use .predict instead of model directly
    shap_values = explainer(X_processed_df)

    os.makedirs("artifacts/shap", exist_ok=True)

    # 🔍 SHAP Summary Bar Plot
    shap.plots.bar(shap_values, show=False)
    plt.savefig("artifacts/shap/shap_summary_bar.png")
    plt.clf()

    # 🔍 SHAP Beeswarm Plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("artifacts/shap/shap_beeswarm.png")
    plt.clf()

    # 🔍 SHAP Waterfall (first sample)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("artifacts/shap/shap_waterfall_sample0.png")
    plt.clf()

    # Display feature name mapping if needed
    print("Top SHAP Features:")
    import pandas as pd
    import numpy as np

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    importance_df = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    print(importance_df.head(5))

    print("✅ SHAP plots saved in artifacts/shap/")
