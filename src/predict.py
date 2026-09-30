import joblib
import pandas as pd


MODEL_PATH = "models/model.pkl"


model = joblib.load(
    MODEL_PATH
)


def predict(data):

    """
    data = dictionary of features
    """

    df = pd.DataFrame(
        [data]
    )


    prediction = model.predict(
        df
    )[0]


    probability = model.predict_proba(
        df
    )[0][1]


    return {

        "prediction": int(prediction),

        "probability": float(probability)

    }