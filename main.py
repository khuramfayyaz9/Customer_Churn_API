from fastapi import FastAPI
from pydantic import BaseModel , Field
from typing import Annotated 
import joblib
import pandas as pd
from scipy.sparse import hstack
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

data = joblib.load("model.pkl")

encoder = data["encoder"]
scaler = data["scaler"]
model = data["model"]




app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get('/about')
def about():
    return{'message':'Customer_Churn_API'}

 # pydantic model to validate data 
class Userinput(BaseModel):
  SeniorCitizen :Annotated[int,Field(...,)]
  tenure :Annotated[int,Field(...,)]
  MonthlyCharges : Annotated[float,Field(...,)]
  TotalCharges : Annotated[float, Field(...)]
  gender : str
  Partner :str
  Dependents : str
  PhoneService : str
  MultipleLines : str
  InternetService : str
  OnlineSecurity : str
  OnlineBackup : str
  DeviceProtection : str
  TechSupport : str
  StreamingTV : str
  StreamingMovies : str
  Contract : str
  PaperlessBilling : str
  PaymentMethod : str

class PredictionResponse(BaseModel):
    prediction: str

#endpoint
@app.post('/predict',response_model=PredictionResponse)
def predict(user: Userinput):
    print("PREDICT ENDPOINT CALLED")
    user_dict=user.model_dump()
    df = pd.DataFrame([user_dict])
    print(df)

    X_cat = df[[
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
 ]]
    
    X_num = df[[
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
 ]]
    X_cat_encoded = encoder.transform(X_cat)
    print(X_cat_encoded.shape)
    X_final = hstack([X_cat_encoded, X_num.values])
    print(X_final.shape)
    X_scaled = scaler.transform(X_final)
    print(X_scaled.shape) 
    prediction = model.predict(X_scaled)[0]
    return {"prediction": prediction}