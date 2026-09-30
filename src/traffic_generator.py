import requests
import random
import time
import csv
import os
from datetime import datetime


API_URL = "http://127.0.0.1:8000/predict"


os.makedirs(
    "logs",
    exist_ok=True
)


file = open(
    "logs/predictions.csv",
    "a",
    newline=""
)


writer = csv.writer(file)


# Create CSV header if file is empty
if os.stat("logs/predictions.csv").st_size == 0:

    writer.writerow([

        "timestamp",

        "age_score",
        "income_score",
        "credit_score",
        "loan_amount",
        "transaction_freq",
        "device_score",
        "location_score",
        "account_age",
        "purchase_score",
        "behavior_score",
        "risk_score",
        "activity_score",

        "prediction",
        "probability"

    ])



while True:


    # Generate fake user traffic
    data = {

        "age_score": random.uniform(0,10),

        "income_score": random.uniform(0,10),

        "credit_score": random.uniform(0,10),

        "loan_amount": random.uniform(0,10),

        "transaction_freq": random.uniform(0,10),

        "device_score": random.uniform(0,10),

        "location_score": random.uniform(0,10),

        "account_age": random.uniform(0,10),

        "purchase_score": random.uniform(0,10),

        "behavior_score": random.uniform(0,10),

        "risk_score": random.uniform(0,10),

        "activity_score": random.uniform(0,10)

    }


    # Send request to FastAPI
    response = requests.post(
        API_URL,
        json=data
    )


    result = response.json()


    print("API RESPONSE:", result)



    # Save only successful predictions
    if "prediction" in result:


        writer.writerow([

            datetime.now(),

            data["age_score"],
            data["income_score"],
            data["credit_score"],
            data["loan_amount"],
            data["transaction_freq"],
            data["device_score"],
            data["location_score"],
            data["account_age"],
            data["purchase_score"],
            data["behavior_score"],
            data["risk_score"],
            data["activity_score"],

            result["prediction"],

            result["probability"]

        ])


        file.flush()


    else:

        print("API Error:", result)



    time.sleep(2)