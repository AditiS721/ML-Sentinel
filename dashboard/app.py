import streamlit as st
import pandas as pd
import plotly.express as px



st.title(
    "🚨 ML Sentinel Dashboard"
)



st.sidebar.header(
    "Model Monitoring"
)



df=pd.read_csv(
    "logs/predictions.csv"
)



# Metrics

st.subheader(
    "Live Predictions"
)


col1,col2,col3=st.columns(3)


col1.metric(

    "Total Requests",

    len(df)

)


col2.metric(

    "Average Risk",

    round(
        df["probability"].mean(),
        3
    )

)


col3.metric(

    "Positive Predictions",

    df["prediction"].sum()

)



# Probability graph
# Probability graph

df["request_id"] = range(
    1,
    len(df)+1
)


fig = px.line(

    df,

    x="request_id",

    y="probability",

    title="Risk Probability Over Time"

)


st.plotly_chart(
    fig,
    use_container_width=True
)



# Prediction distribution

fig2 = px.histogram(

    df,

    x="prediction",

    title="Prediction Distribution",

    labels={
        "prediction":"Prediction Class"
    }

)


st.plotly_chart(
    fig2,
    use_container_width=True
)