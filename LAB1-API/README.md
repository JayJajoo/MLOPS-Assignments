# XGBoost Prediction API

This project provides a FastAPI-based API for predicting outcomes using a pre-trained XGBoost model.

## Installation

1. Clone the repository:

- git clone https://github.com/JayJajoo/MLOPS-Assignments.git
- cd .\LAB1-API\

2. Install the required packages:

- pip install -r ./requirements.txt

## Execution

1. Run the training script:

- python train.py

2. Start the FastAPI server:

- uvicorn app:app --reload

- Server will run at: http://127.0.0.1:8000

## Testing the API

1. Open the interactive API docs:

- http://127.0.0.1:8000/docs

2. Find the Predict API endpoint and click "Try it Out".

3. Paste the following JSON example into the request body:

- {
  "mean radius": 12.47,
  "mean texture": 18.6,
  "mean perimeter": 81.09,
  "mean area": 481.9,
  "mean smoothness": 0.09965,
  "mean compactness": 0.1058,
  "mean concavity": 0.08005,
  "mean concave points": 0.03821,
  "mean symmetry": 0.1925,
  "mean fractal dimension": 0.06373,
  "radius error": 0.3961,
  "texture error": 1.044,
  "perimeter error": 2.497,
  "area error": 30.29,
  "smoothness error": 0.006953,
  "compactness error": 0.01911,
  "concavity error": 0.02701,
  "concave points error": 0.01037,
  "symmetry error": 0.01782,
  "fractal dimension error": 0.003586,
  "worst radius": 14.97,
  "worst texture": 24.64,
  "worst perimeter": 96.05,
  "worst area": 677.9,
  "worst smoothness": 0.1426,
  "worst compactness": 0.2378,
  "worst concavity": 0.2671,
  "worst concave points": 0.1015,
  "worst symmetry": 0.3014,
  "worst fractal dimension": 0.0875
}

4. Click "Execute" to get the prediction.

- Expected Outcome:

- {
  "prediction": "Person is Diabetic"
}
