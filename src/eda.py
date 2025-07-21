# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('Lead Scoring.csv')

# Identifies numerical and categorical columns, excluding ID columns
def get_column_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['Prospect ID', 'Lead Number']]
    return num_cols, cat_cols

# Displays the count and percentage of missing values in each column
def missing_value_report(df):
    nulls = df.isnull().sum().to_frame(name='Missing Count')
    nulls['Missing %'] = (nulls['Missing Count'] / len(df)) * 100
    nulls = nulls[nulls['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    print("Columns with missing values:")
    print(nulls)

# Displays summary statistics (mean, median, std, etc.) for each numerical feature
def univariate_numerical(df):
    num_cols, _ = get_column_types(df)
    for col in num_cols:
        print(f"\nSummary: {col}")
        print(df[col].describe().round(2))

# Displays frequency distribution for each categorical feature
def univariate_categorical(df):
    _, cat_cols = get_column_types(df)
    for col in cat_cols:
        print(f"\nValue Counts: {col}")
        print(df[col].value_counts(dropna=False).head(10))

# Displays the average target conversion rate for each category in categorical columns
def categorical_vs_target_rate(df, cat_cols, target='Converted'):
    for col in cat_cols:
        if df[col].nunique() <= 20:
            print(f"\nConversion Rate by {col}:")
            print(df.groupby(col)[target].mean().sort_values(ascending=False).round(2))

# Displays grouped descriptive statistics of numerical columns with respect to the target variable
def numerical_vs_target_stats(df, num_cols, target='Converted'):
    for col in num_cols:
        print(f"\nStatistics of {col} grouped by {target}:")
        print(df.groupby(target)[col].describe().round(2))

# Displays the correlation matrix between numerical columns
def correlation_matrix(df):
    num_cols, _ = get_column_types(df)
    print("Correlation Matrix:")
    print(df[num_cols].corr().round(2))

# Plots the distribution curve for each numerical feature
def plot_numerical_distribution(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='steelblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# Plots the top 10 categories by frequency for each categorical feature
def plot_categorical_distribution(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(14, 6))
        order = df[col].value_counts().iloc[:10].index
        sns.countplot(data=df, x=col, order=order, palette='viridis')
        plt.title(f'Top Categories in {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Plots the conversion rate for categories in low-cardinality categorical columns
def plot_conversion_rate(df, cat_cols, target='Converted'):
    for col in cat_cols:
        if df[col].nunique() <= 20:
            plt.figure(figsize=(14, 6))
            conv_rate = df.groupby(col)[target].mean().sort_values(ascending=False)
            sns.barplot(x=conv_rate.index.astype(str), y=conv_rate.values, palette='coolwarm')
            plt.title(f'Conversion Rate by {col}')
            plt.ylabel('Conversion Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Plots boxplots to compare numerical features against the target variable
def plot_numerical_vs_target(df, num_cols, target='Converted'):
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=target, y=col, data=df, palette='Set2')
        plt.title(f'{col} by {target}')
        plt.tight_layout()
        plt.show()

# Plots the heatmap of correlation matrix for numerical features
def plot_correlation_matrix(df, num_cols):
    corr = df[num_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

# Plots a heatmap showing locations of missing values across the dataset
def plot_missing_heatmap(df):
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='YlGnBu')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()

# Identifies outliers in numerical columns using z-score method
def detect_outliers(df, num_cols, threshold=3):
    outlier_report = {}
    for col in num_cols:
        zs = zscore(df[col].dropna())
        outliers = np.where(np.abs(zs) > threshold)[0]
        outlier_report[col] = len(outliers)
    return outlier_report

# Plots the class distribution of the target variable
def plot_target_distribution(df, target='Converted'):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, palette='pastel')
    plt.title('Target Variable Distribution')
    plt.tight_layout()
    plt.show()

# Identifies columns that contain placeholders such as "Select", "Unknown", or "Other"
def check_placeholder_values(df, placeholder_list=['Select', 'Unknown', 'Other']):
    for col in df.columns:
        if df[col].dtype == 'object':
            values = df[col].astype(str).unique()
            for placeholder in placeholder_list:
                if placeholder in values:
                    print(f"'{placeholder}' found in column: {col}")


def exploratory_data_analysis(df):
    print("\n--- Step 1: Detecting Column Types ---")
    num_cols, cat_cols = get_column_types(df)
    print(f"Numerical Columns: {num_cols}")
    print(f"Categorical Columns: {cat_cols}")

    print("\n--- Step 2: Missing Value Report ---")
    missing_value_report(df)
    plot_missing_heatmap(df)

    print("\n--- Step 3: Univariate Analysis for Numerical Features ---")
    univariate_numerical(df)
    plot_numerical_distribution(df, num_cols)

    print("\n--- Step 4: Univariate Analysis for Categorical Features ---")
    univariate_categorical(df)
    plot_categorical_distribution(df, cat_cols)

    print("\n--- Step 5: Target Variable Distribution ---")
    plot_target_distribution(df)

    print("\n--- Step 6: Conversion Rate by Categorical Variables ---")
    categorical_vs_target_rate(df, cat_cols)

    print("\n--- Step 7: Numerical Statistics by Target ---")
    numerical_vs_target_stats(df, num_cols)

    print("\n--- Step 8: Visualizing Numerical Features vs Target ---")
    plot_numerical_vs_target(df, num_cols)

    print("\n--- Step 9: Correlation Analysis ---")
    correlation_matrix(df)
    plot_correlation_matrix(df, num_cols)

    print("\n--- Step 10: Conversion Rate Plot for Categorical Variables ---")
    plot_conversion_rate(df, cat_cols)

    print("\n--- Step 11: Outlier Detection ---")
    outliers = detect_outliers(df, num_cols)
    print("Outliers detected (z-score > 3):")
    for col, count in outliers.items():
        print(f"{col}: {count} outliers")

    print("\n--- Step 12: Placeholder Value Check ---")
    check_placeholder_values(df)

