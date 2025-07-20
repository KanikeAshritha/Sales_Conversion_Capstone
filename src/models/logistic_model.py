from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n🔍 Logistic Regression Report:")
    print(classification_report(y_test, preds))
    return model
