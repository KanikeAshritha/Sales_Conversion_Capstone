from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

def train_lightgbm(X_train, y_train, X_test, y_test):
    model = LGBMClassifier(random_state=42,verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n💡 LightGBM Report:")
    print(classification_report(y_test, preds))
    return model
