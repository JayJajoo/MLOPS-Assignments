from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel, Field

app = FastAPI()

class InputFormat(BaseModel):
    mean_radius: float = Field(..., alias='mean radius')
    mean_texture: float = Field(..., alias="mean texture")
    mean_perimeter: float = Field(..., alias="mean perimeter")
    mean_area: float = Field(..., alias="mean area")
    mean_smoothness: float = Field(..., alias="mean smoothness")
    mean_compactness: float = Field(..., alias="mean compactness")
    mean_concavity: float = Field(..., alias="mean concavity")
    mean_concave_points: float = Field(..., alias="mean concave points")
    mean_symmetry: float = Field(..., alias="mean symmetry")
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension")
    radius_error: float = Field(..., alias="radius error")
    texture_error: float = Field(..., alias="texture error")
    perimeter_error: float = Field(..., alias="perimeter error")
    area_error: float = Field(..., alias="area error")
    smoothness_error: float = Field(..., alias="smoothness error")
    compactness_error: float = Field(..., alias="compactness error")
    concavity_error: float = Field(..., alias="concavity error")
    concave_points_error: float = Field(..., alias="concave points error")
    symmetry_error: float = Field(..., alias="symmetry error")
    fractal_dimension_error: float = Field(..., alias="fractal dimension error")
    worst_radius: float = Field(..., alias="worst radius")
    worst_texture: float = Field(..., alias="worst texture")
    worst_perimeter: float = Field(..., alias="worst perimeter")
    worst_area: float = Field(..., alias="worst area")
    worst_smoothness: float = Field(..., alias="worst smoothness")
    worst_compactness: float = Field(..., alias="worst compactness")
    worst_concavity: float = Field(..., alias="worst concavity")
    worst_concave_points: float = Field(..., alias="worst concave points")
    worst_symmetry: float = Field(..., alias="worst symmetry")
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension")

with open("./model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def greet():
    return "Welcome to Diabetes Prediction Portal"

@app.post("/")
def predict(model_input: InputFormat):
    try:
        data = model_input.model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Input Format Not Correct.")
    vals = np.array(list(data.values())).reshape(1, -1)
    y_pred = int(model.predict(vals)[0])
    if y_pred==1:
        return {"prediction":"patient is diabetic."}
    return {"prediction":"patient is not diabetic."}
