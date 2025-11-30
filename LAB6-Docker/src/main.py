# Dockerized ML Training Pipeline
# This script trains two different models on a chosen dataset (Wine dataset)
# and compares their accuracy. It is structured to be used inside a Docker container.

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")


def train_models():
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    joblib.dump(rf, "/app/models/wine_rf_model.pkl")

    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    joblib.dump(lr, "/app/models/wine_lr_model.pkl")

    print("Random Forest Accuracy:", rf_acc)
    print("Logistic Regression Accuracy:", lr_acc)
    print("Better Model:", "Random Forest" if rf_acc > lr_acc else "Logistic Regression")

if __name__ == "__main__":
    train_models()
