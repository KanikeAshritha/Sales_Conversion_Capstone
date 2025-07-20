import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_csv('Lead Scoring.csv')

def get_column_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['Prospect ID', 'Lead Number']]
    return num_cols, cat_cols

def missing_value_report(df):
    nulls = df.isnull().sum().to_frame(name='Missing Count')
    nulls['Missing %'] = (nulls['Missing Count'] / len(df)) * 100
    nulls = nulls[nulls['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    print("Columns with missing values:")
    print(nulls)

def univariate_numerical(df):
    num_cols, _ = get_column_types(df)
    for col in num_cols:
        print(f"\nSummary: {col}")
        print(df[col].describe().round(2))

def univariate_categorical(df):
    _, cat_cols = get_column_types(df)
    for col in cat_cols:
        print(f"\nValue Counts: {col}")
        print(df[col].value_counts(dropna=False).head(10))

def categorical_vs_target_rate(df, cat_cols, target='Converted'):
    for col in cat_cols:
        if df[col].nunique() <= 20:
            print(f"\nConversion Rate by {col}:")
            print(df.groupby(col)[target].mean().sort_values(ascending=False).round(2))

def numerical_vs_target_stats(df, num_cols, target='Converted'):
    for col in num_cols:
        print(f"\nStatistics of {col} grouped by {target}:")
        print(df.groupby(target)[col].describe().round(2))

def correlation_matrix(df):
    num_cols, _ = get_column_types(df)
    print("Correlation Matrix:")
    print(df[num_cols].corr().round(2))

def plot_numerical_distribution(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='steelblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

def plot_categorical_distribution(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(14, 6))
        order = df[col].value_counts().iloc[:10].index
        sns.countplot(data=df, x=col, order=order, palette='viridis')
        plt.title(f'Top Categories in {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

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

def plot_numerical_vs_target(df, num_cols, target='Converted'):
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=target, y=col, data=df, palette='Set2')
        plt.title(f'{col} by {target}')
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix(df, num_cols):
    corr = df[num_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def plot_missing_heatmap(df):
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='YlGnBu')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()

def detect_outliers(df, num_cols, threshold=3):
    outlier_report = {}
    for col in num_cols:
        zs = zscore(df[col].dropna())
        outliers = np.where(np.abs(zs) > threshold)[0]
        outlier_report[col] = len(outliers)
    return outlier_report


def plot_target_distribution(df, target='Converted'):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, palette='pastel')
    plt.title('Target Variable Distribution')
    plt.tight_layout()
    plt.show()


def check_placeholder_values(df, placeholder_list=['Select', 'Unknown', 'Other']):
    _, cat_cols = get_column_types(df)
    print("\nPlaceholder Counts:")
    for col in cat_cols:
        counts = df[col].isin(placeholder_list).sum()
        if counts > 0:
            print(f"{col}: {counts} placeholder values")

def detect_rare_categories(df, cat_cols, threshold=0.01):
    print("\nRare Category Report (less than 1% frequency):")
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < threshold]
        if not rare.empty:
            print(f"{col}:\n{rare.round(4)}\n")

def missingness_vs_target(df, cat_cols, target='Converted'):
    print("\nConversion Rate Based on Missingness:")
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            print(f"{col}:")
            print(df.groupby(f'{col}_missing')[target].mean().round(2))


num_cols, cat_cols = get_column_types(df)

missing_value_report(df)
plot_missing_heatmap(df)

univariate_numerical(df)
univariate_categorical(df)

categorical_vs_target_rate(df, cat_cols)
numerical_vs_target_stats(df, num_cols)
correlation_matrix(df)

plot_target_distribution(df)
plot_numerical_distribution(df, num_cols)
plot_categorical_distribution(df, cat_cols)
plot_conversion_rate(df, cat_cols)
plot_numerical_vs_target(df, num_cols)
plot_correlation_matrix(df, num_cols)
plot_top_correlated_pairplot(df, num_cols)

outlier_counts = detect_outliers(df, num_cols)
print("\nOutlier counts per numerical column:")
print(outlier_counts)

check_placeholder_values(df)
detect_rare_categories(df, cat_cols)
missingness_vs_target(df, cat_cols)
