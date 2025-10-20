import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage
import joblib
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import sys

# --- Step 1: Download Iris data ---
def download_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target)
    return features, target

# --- Step 2: Split data ---
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# --- Step 3: Train model ---
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Step 4: Save model to GCS ---
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename("model.joblib")

    print(f"Model uploaded to GCS: gs://{bucket_name}/{blob_name}")

# --- Step 5: Send classification report via email ---
def send_email_report(report_text, accuracy, recipient_email):
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")

    if not sender_email or not sender_password:
        raise ValueError("EMAIL_USER and EMAIL_PASS environment variables must be set")

    msg = EmailMessage()
    msg["Subject"] = "Random Forest Classification Report"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    msg.set_content(
        f"""
          Hello,

          Here is the classification report for your trained RandomForest model:

          Overall accuracy: {accuracy:.4f}

          Classification report:
          {report_text}

          Model training and upload completed successfully.

          Regards,
          Your ML Pipeline
          """
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

    print(f"Email sent to {recipient_email}")

# --- Step 6: Main workflow ---
def main():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"🔹 Model accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", report)

    # --- Accuracy threshold check ---
    threshold = 0.75
    if accuracy < threshold:
        print(f"Accuracy below threshold ({threshold * 100}%). Skipping upload and email.")
        sys.exit(1)  # Cause GitHub Actions to stop

    print("✅ Accuracy threshold met. Proceeding with upload and email...")

    # --- Save model to GCS ---
    bucket_name = "mlops_assignment_bucket_1"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/model_{timestamp}.joblib"
    save_model_to_gcs(model, bucket_name, blob_name)

    # --- Send email report ---
    recipient_email = "jayjajoo02@example.com"
    send_email_report(report, accuracy, recipient_email)


if __name__ == "__main__":
    main()



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from google.cloud import storage
# import joblib
# from datetime import datetime

# # Download necessary data - Iris data from sklearn library
# # We define a function to download the data
# def download_data():
#   from sklearn.datasets import load_iris
#   iris = load_iris()
#   features = pd.DataFrame(iris.data, columns=iris.feature_names)
#   target = pd.Series(iris.target)
#   return features, target

# # Define a function to preprocess the data
# # In this case, preprocessing will be just splitting the data into training and testing sets
# def preprocess_data(X, y):
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#   return X_train, X_test, y_train, y_test

# # Define a function to train the model
# def train_model(X_train, y_train):
#   model = RandomForestClassifier(n_estimators=100, random_state=42)
#   model.fit(X_train, y_train)
#   return model

# # Define a function to save the model both locally and in GCS
# def save_model_to_gcs(model, bucket_name, blob_name):
#   joblib.dump(model, "model.joblib")
  
#   # Save the model to GCS
#   storage_client = storage.Client()
#   bucket = storage_client.bucket(bucket_name)
#   blob = bucket.blob(blob_name)
#   blob.upload_from_filename('model.joblib')

# # Putting all functions together
# def main():
#   # Download data
#   X, y = download_data()
#   X_train, X_test, y_train, y_test = preprocess_data(X, y)
  
#   # Train model
#   model = train_model(X_train, y_train)
  
#   # Evaluate model
#   y_pred = model.predict(X_test)
#   accuracy = accuracy_score(y_test, y_pred)
#   print(f'Model accuracy: {accuracy}')
  
#   # Save the model to gcs
#   bucket_name = "mlops_assignment_bucket_1"
#   timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#   blob_name = f"trained_models/model_{timestamp}.joblib"
#   save_model_to_gcs(model, bucket_name, blob_name)
#   print(f"Model saved to gs://{bucket_name}/{blob_name}")
  
# if __name__ == "__main__":
#   main()

  
