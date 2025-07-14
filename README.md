# **📝 California Housing Price Prediction**
## 🔖 Team Name: DeciForge
## 📌 Project Title: California Housing Price Analysis and Prediction Using Machine Learning
### 🔍 1. Introduction
This project, undertaken by Team DeciForge, explores key insights from the California housing dataset using data cleaning, feature engineering, and visual analytics. The goal is to identify and understand factors that influence house prices such as income, age of housing, population density, and proximity to the ocean.

### 📚 2. Data Cleaning & Preparation
- Loaded data from cleaned.csv.
- Removed rows with missing values in the total_bedrooms column to ensure consistency and reliability.
- Verified column names and data types for smooth processing.

### 🏗️ 3. Feature Engineering
- Area Density: Computed as population divided by number of households.
- Income Category: Grouped median income into 'High' and 'Low' categories based on a threshold value of 6.
- Age Group: Categorized houses into 'New' (≤15 years), 'Middle-aged' (16–35 years), and 'Old' (>35 years).
- Rooms per Household: Derived to evaluate average space availability.
- Capped Age Flag: Identified homes with the maximum age value of 52.
- Ocean Proximity Encoding: Transformed the categorical ocean_proximity column into dummy variables for analysis.

### 📊 4. Visual Analysis & Insights
🔸 Univariate Analysis
- Histogram of median_house_value revealed a right-skewed distribution and a price ceiling around $500,000.
- Pie chart displayed proportions of income categories.
- Donut chart showed the distribution of housing age groups.

🔸 Bivariate Analysis
- Bar plots illustrated:
- Average house values by income category (High income areas had significantly higher house values).
- Price variation among housing age groups, with older homes generally.

🔸 Multivariate Analysis
- Heatmap of numerical features revealed strong positive correlation between median_income and median_house_value.
- Box plots helped detect outliers in features like total_rooms, population, and households.
- Scatter plots between income and house value confirmed increasing trends with income rise.

###  💻 5. Machine Learning Modeling
Trained and evaluated the following models using scaled and preprocessed data:
* Linear Regression: Basic baseline model to establish linear trend.
* Decision Tree Regressor: Captured non-linearities but prone to overfitting.
* Random Forest Regressor: Most accurate with lowest RMSE; also used to derive feature importance.

### 🎯 Performance Evaluation
- Root Mean Squared Error (RMSE) was calculated for each model to measure prediction accuracy.
- Visualized Actual vs Predicted plots to better understand the model fit on test data.
  
### 📌 Additional Insights from ML
* Linear regression with a single feature (median_income) demonstrated a clear income-house value relationship.
* Feature importance plot from Random Forest identified top contributing features like median_income, area_density, and total_rooms.

###  🗂️ 6. Power BI Dashboard
An interactive Power BI dashboard was created for visual storytelling and business understanding:
* 📊 KPIs like average housing age, median income, and total population.
* 📌 Geo-scatter map shows population concentration across California.
* 🧩 Filters allow segmentation by age category and ocean proximity.
* 🟣 Doughnut charts reveal distribution of houses and rooms based on proximity to ocean.
* 📈 Dynamic bar and table views track median house values by multiple factors.

### 🚀 7. Future Work
* Incorporate More Features:
- Integrate environmental indicators, crime rates, school ratings, or job market indices to improve prediction accuracy.

* Time Series Forecasting:
- Add a temporal dimension using past housing prices to forecast future trends using models like ARIMA, LSTM, or Prophet.

* Model Deployment:
- Convert the trained model into a deployable API using Flask or FastAPI, and integrate it into a web app for public use.

* Interactive Web Dashboard:
- Upgrade from Power BI to a Plotly Dash or Streamlit app for interactive, code-integrated insights and live prediction functionality.

* Clustering Analysis:
- Use K-Means or DBSCAN to segment neighborhoods/regions into distinct housing markets based on price, density, and proximity factors.

