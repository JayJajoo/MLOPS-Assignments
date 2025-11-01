"""
Simple Inference Script - Load registered model and predict on single sample
"""

import mlflow.sklearn
import numpy as np

# Configuration
MODEL_NAME = "iris_best_model"
STAGE = "Production"  # or use version number like "1"

# Load model from registry
print(f"Loading model: {MODEL_NAME} from {STAGE} stage...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")
print("✓ Model loaded successfully\n")

# Sample input (sepal_length, sepal_width, petal_length, petal_width)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# Make prediction
prediction = model.predict(sample_input)
probabilities = model.predict_proba(sample_input)

# Display result
class_names = ['setosa', 'versicolor', 'virginica']
predicted_class = class_names[prediction[0]]

print("Input:")
print(f"  Sepal Length: {sample_input[0][0]}")
print(f"  Sepal Width:  {sample_input[0][1]}")
print(f"  Petal Length: {sample_input[0][2]}")
print(f"  Petal Width:  {sample_input[0][3]}")

print(f"\nPredicted Class: {predicted_class}")

print("\nClass Probabilities:")
for class_name, prob in zip(class_names, probabilities[0]):
    print(f"  {class_name:15s}: {prob:.4f} ({prob*100:.2f}%)")