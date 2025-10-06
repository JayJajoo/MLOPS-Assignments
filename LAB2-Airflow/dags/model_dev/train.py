import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DATA_PATH = "/opt/airflow/dags/model_dev/data"
MODEL_DIR = "/opt/airflow/dags/models"

def load_data():
    X, Y = load_iris(return_X_y=True, as_frame=True)
    X["Target"] = Y
    X.to_csv(os.path.join(DATA_PATH, "raw_data.csv"), index=False)
    print("Raw data saved.")

def split_dataset():
    df = pd.read_csv(os.path.join(DATA_PATH, "raw_data.csv"))
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1], train_size=0.8, random_state=42
    )
    X_train.to_csv(os.path.join(DATA_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PATH, "y_test.csv"), index=False)
    print("Data split completed.")

def train_model():
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print("Model trained.")

    # Save the model
    model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved at {model_path}")

    return model_path

def evaluate_model(model_path):
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))

    # Load the model
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)
    print("Evaluation report:\n", report)
    return report

def send_email(report, sender_email, receiver_email, smtp_server, smtp_port, login, password):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "Model Evaluation Report"

    msg.attach(MIMEText(report, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(login, password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)
