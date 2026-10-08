from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # Add CORS middleware
from fastapi.staticfiles import StaticFiles


ml_model = {} # {'model': "credit_risk_model", 'threshold': threshold}



@asynccontextmanager #decorator to manage the lifespan of the FastAPI app
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load('credit_risk_model.pkl')  # Load the model
    ml_model['threshold'] = joblib.load('best_threshold.pkl')  # Load the threshold
    
    yield
    
    ml_model.clear()  # Clear the model from memory when the app shuts down
    
app = FastAPI(lifespan=lifespan)  # Use the lifespan context manager
    
    
# Input data model for loan application so no loan_status    
class LoanApplication(BaseModel): #pydantic model for input data validation
    person_age:             int
    person_income:          int
    person_home_ownership:  str
    person_emp_length:      float
    loan_intent:            str
    loan_grade:             str
    loan_amnt:              int
    loan_int_rate:          float
    loan_percent_income:    float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int



# @app.get("/")
# def greet():
#     return {"message": "Hello, World!"}




@app.post("/predict/")
def predict(loan_application: LoanApplication):
    input_df = pd.DataFrame([loan_application.dict()])  # Convert dictionary to DataFrame
    # Get the probability of the positive class
    prediction_proba = ml_model['model'].predict_proba(input_df)[:, 1][0]  
    
    prediction = int(prediction_proba >= ml_model['threshold'])
    return {
        "default_probability": prediction_proba,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")  # Serve static files