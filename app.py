import requests

url = "https://disease.sh/v3/covid-19/countries/Ireland"
r = requests.get(url)
data = r.json()

print(data)

import pandas as pd

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

import matplotlib.pyplot as plt

labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()




# Predict next day's cases


import numpy as np

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR



X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mod = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
mod.fit(X_train, y_train)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
predicted_cases = mod.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

predicted_cases = model.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

import streamlit as st

st.title("COVID-19 Cases Prediction-in IRELAND")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = mod.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df_historical["day"], df_historical["cases"], label="Cases", color="blue", marker="o")
ax.plot(df_historical["day"], df_historical["deaths"], label="Deaths", color="red", marker="x")
ax.set_xlabel("Day")
ax.set_ylabel("Count")
ax.set_title("Historical COVID-19 Cases and Deaths (Last 30 Days)")
ax.legend()
st.pyplot(fig)
