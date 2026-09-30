from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os


# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.predict import predict



app = FastAPI(
    title="ML Sentinel API",
    description="Real-time ML Monitoring System"
)



class CustomerInput(BaseModel):

    age_score: float

    income_score: float

    credit_score: float

    loan_amount: float

    transaction_freq: float

    device_score: float

    location_score: float

    account_age: float

    purchase_score: float

    behavior_score: float

    risk_score: float

    activity_score: float







@app.get("/")
def home():

    return {
        "message": "ML Sentinel API running"
    }



@app.post("/predict")
def prediction(
    data: CustomerInput
):

    result = predict(
        data.dict()
    )

    return result