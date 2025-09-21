from sklearn.datasets import load_breast_cancer
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

X,Y = load_breast_cancer(return_X_y=True,as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

xgb = xgboost.XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

clf = classification_report(y_test,y_pred)
print(clf)

cnf = confusion_matrix(y_test,y_pred)
cmd = ConfusionMatrixDisplay(cnf)
cmd.plot()
plt.savefig("confusion_matrix.png")
plt.show()

with open("./model/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)
