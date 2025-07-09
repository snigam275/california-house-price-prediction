# **📝 California Housing Price Prediction**
## 🔖 Team Name: DeciForge
## 📌 Project Title: California Housing Price Analysis and Prediction Using Machine Learning
### 🔍 1. Introduction
This project, undertaken by Team DeciForge, explores key insights from the California housing dataset using data cleaning, feature engineering, and visual analytics. The goal is to identify and understand factors that influence house prices such as income, age of housing, population density, and proximity to the ocean.

### 2. Data Cleaning & Preparation
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


