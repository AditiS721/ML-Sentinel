import pandas as pd

from evidently.report import Report

from evidently.metric_preset import DataDriftPreset



reference = pd.read_csv(
    "data/train.csv"
)


current = pd.read_csv(
    "logs/predictions.csv"
)



current=current.drop(

    columns=[
        "timestamp",
        "prediction",
        "probability"
    ]

)



report = Report(

    metrics=[

        DataDriftPreset()

    ]

)



report.run(

    reference_data=reference.drop(
        columns=["target"]
    ),

    current_data=current

)



report.save_html(

    "drift_report.html"

)


print(
    "Drift report generated"
)