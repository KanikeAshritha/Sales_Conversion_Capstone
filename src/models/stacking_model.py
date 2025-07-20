from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_stacking_model(X_train, y_train, X_test, y_test):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n🧠 Stacking Classifier Report:")
    print(classification_report(y_test, preds))
    return model
