# Dockerized ML Training Pipeline

This project demonstrates how to train two machine learning models inside a Docker container using the Wine dataset from scikit-learn. At the end of training, the script compares both models and prints their accuracies.

---

## 📁 Folder Structure
```
LAB6-Docker/
│
├── Dockerfile
├── README.md
└── src/
    ├── main.py
    └── requirements.txt
```

---

## 🚀 What This Project Does
- Loads the **Wine dataset** from scikit-learn
- Trains **two models**:
  - RandomForestClassifier
  - LogisticRegression
- Saves both models as `.pkl` files
- Compares their accuracy on the test set
- Runs entirely inside a **Docker container**

---

## 🧠 Models Trained
| Model | Description | Output File |
|-------|-------------|-------------|
| **Random Forest** | Ensemble of decision trees | `wine_rf_model.pkl` |
| **Logistic Regression** | Linear classifier | `wine_lr_model.pkl` |

The script prints:
- Accuracy of each model
- Which model performed better

---

## 📦 How to Build and Run the Docker Container

### 1. Build the Docker image
```bash
docker build -t wine-trainer .
```

### 2. Run the container
```bash
docker run --rm wine-trainer
```

Model files will be stored **inside the container** unless you mount a volume.

To save models to your host machine:
```bash
docker run --rm -v "${PWD}\models:/app/models" wine-trainer
```

---

## 📜 main.py Overview
The script:
1. Loads dataset
2. Splits into train and test
3. Trains RandomForest & LogisticRegression
4. Saves both models
5. Prints accuracy comparison

---

## 📄 requirements.txt
All Python dependencies are listed in `requirements.txt` and installed inside Docker during build.

---

## 🐳 Dockerfile Overview
The Dockerfile typically:
- Uses Python base image
- Copies project files
- Installs dependencies
- Runs `main.py`

---