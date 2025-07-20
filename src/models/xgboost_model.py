from xgboost import XGBClassifier
from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(eval_metric='logloss', random_state=42,verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n⚡ XGBoost Report:")
    print(classification_report(y_test, preds))
    return model
