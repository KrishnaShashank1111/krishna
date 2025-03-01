import requests
import pandas as pd
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Fetch COVID-19 data for China
url = "https://disease.sh/v3/covid-19/countries/China"
r = requests.get(url)
data = r.json()

print(data)

# Extract relevant fields for visualization and regression
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

# Visualization: Bar plot of total cases, active cases, recovered, and deaths
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for China")
plt.show()

# Generate random historical data for prediction
random.seed(42)
historical_cases = [random.randint(30000, 70000) for _ in range(30)]  # Last 30 days cases
historical_deaths = [random.randint(500, 2000) for _ in range(30)]

# Create DataFrame
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Features (day) and target (cases/deaths)
X = df_historical["day"].values.reshape(-1, 1)  # Day as feature
y_cases = df_historical["cases"].values  # Cases as target
y_deaths = df_historical["deaths"].values  # Deaths as target

# Initialize and train the SVR model for cases
svr_cases = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1)
svr_cases.fit(X, y_cases)

# Predict using the trained model
predicted_cases = svr_cases.predict(X)

# Initialize and train the SVR model for deaths
svr_deaths = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1)
svr_deaths.fit(X, y_deaths)

# Predict using the trained model
predicted_deaths = svr_deaths.predict(X)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot for cases
plt.subplot(1, 2, 1)
plt.scatter(X, y_cases, color='blue', label='Actual Cases')
plt.plot(X, predicted_cases, color='red', label='Predicted Cases')
plt.title('SVR Regression for Cases')
plt.xlabel('Day')
plt.ylabel('Cases')
plt.legend()

# Plot for deaths
plt.subplot(1, 2, 2)
plt.scatter(X, y_deaths, color='green', label='Actual Deaths')
plt.plot(X, predicted_deaths, color='orange', label='Predicted Deaths')
plt.title('SVR Regression for Deaths')
plt.xlabel('Day')
plt.ylabel('Deaths')
plt.legend()

plt.tight_layout()
plt.show()

# Print predicted values for the last day (Day 30)
print(f"Predicted cases for day 30: {predicted_cases[-1]}")
print(f"Predicted deaths for day 30: {predicted_deaths[-1]}")

# Streamlit interactive app for predicting future cases
st.title("COVID-19 Cases Prediction for China")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User input for the day number
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = svr_cases.predict([[day_input]])  # Using SVM model to predict cases for the given day
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
