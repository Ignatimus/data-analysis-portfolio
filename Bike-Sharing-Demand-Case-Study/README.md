# 🚲 Predicting Hourly Bike-Sharing Demand in Seoul

**By Anatoli Ignatov | May 2026**

[Jupyter Notebook Binder](https://mybinder.org/)  

## 🗂️ About the repo
This folder contains a `.ipynb` **Jupyter Notebook** and supporting materials for an end-to-end data science project focused on predicting hourly bike-sharing demand in Seoul, South Korea. Inside the notebook you will find:

1. **Project Summary** — Dataset background, business objective, and project workflow
2. **Data Collection** — Web scraping with `rvest` and API data collection with `httr`
3. **Data Cleaning & Wrangling** — Regular expressions, feature engineering, normalization, and missing-value handling
4. **SQL Exploratory Analysis** — SQL queries for seasonality, weather patterns, and bike demand trends
5. **Visual Exploratory Analysis** — Statistical visualizations built with `ggplot2`
6. **Regression Modeling** — Baseline and refined linear regression models with polynomial features, interaction terms, and regularization
7. **Final Reflections** — Key findings, limitations, and lessons learned

Additionally, there is a **Visualizations** folder containing exported plots and charts generated throughout the analysis.

This **README** also includes a [Binder link](https://mybinder.org/) to launch the notebook interactively in the browser. For easier navigation inside the notebook, use the **“Table of Contents”** panel on the left side.

---

## 📎 Dataset
The primary dataset used in this project is the **Seoul Bike Sharing Demand Dataset**, publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand).

The dataset contains hourly bike rental information together with weather and seasonal variables.

Variable | Description |
-----|-----|
Date | Observation date |
Rented Bike Count | Number of rented bikes per hour |
Hour | Hour of the day |
Temperature | Temperature in °C |
Humidity | Humidity percentage |
Wind speed | Wind speed in m/s |
Visibility | Visibility distance |
Dew point temperature | Dew point temperature in °C |
Solar Radiation | Solar radiation level |
Rainfall | Rainfall amount |
Snowfall | Snowfall amount |
Seasons | Winter, Spring, Summer, Autumn |
Holiday | Whether the day is a holiday |
Functioning Day | Whether the bike-sharing system operated normally |

Additional external data was collected through:
- **Wikipedia web scraping** for global bike-sharing system information
- **OpenWeather API** for live weather forecast data

---

## 🛠️ Tools Used
* JupyterLab
   * R
      * tidyverse
      * dplyr
      * ggplot2
      * stringr
      * rvest
      * httr
      * readr
      * RSQLite
      * tidymodels
      * glmnet

---

## ❓ Business Problem
The objective of this project is to answer the following question:

**Can hourly bike-sharing demand in Seoul be predicted accurately using weather, seasonal, and time-based variables?**

Accurate demand forecasting can help bike-sharing systems:
- Improve bike distribution efficiency
- Reduce shortages and overcrowding
- Optimize operational planning
- Better prepare for seasonal and weather-related demand fluctuations

---

## 💡 Key Insights from Data Analysis
1. Bike demand strongly increases during warmer temperatures and favorable weather conditions.
2. Rental activity follows a clear hourly cycle, with peak demand occurring during commuting hours.
3. Summer and autumn showed the highest average rental counts.
4. Rainfall and snowfall significantly reduce bike demand.
5. Bike rentals are substantially lower during winter months.
6. Weather variables alone were not sufficient for strong predictive performance — time and seasonal variables improved the model considerably.
7. Polynomial and interaction features improved model fit by capturing nonlinear relationships in the data.

---

## 🧠 Model Development and Evaluation
Several regression models were developed and compared throughout the project:

* **Baseline Linear Regression (Weather Variables Only)** — Limited predictive power.
* **Expanded Linear Regression** — Included time, seasonality, and categorical variables.
* **Polynomial Regression** — Captured nonlinear relationships between weather and demand.
* **Interaction-Term Regression** — Improved performance by modeling combined variable effects.
* **Regularized Regression with glmnet** — Reduced overfitting and improved generalization.

Model evaluation included:
- RMSE
- R²
- Residual analysis
- Q-Q calibration plots
- Comparative visualization of experiments

The refined regression models significantly outperformed the baseline specifications.

---

## PART 2 WORK IN PROGRESS

---
