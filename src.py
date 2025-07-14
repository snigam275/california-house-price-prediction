### Libraries Used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Loading of dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Desktop\\summer_term\\Project\\pro\\housing.csv")

# Cleaning of dataset
print(df)
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

df["total_bedrooms"] = df["total_bedrooms"].dropna()

df['area_density'] = df['population']/df['households']
print(df['area_density'])

income_category = []
for i in df['median_income']:
    if i<=4:
        income_category.append("Low Income")
    elif i>4 and i<=6.5:
        income_category.append("Moderate Income")
    else:
        income_category.append("High Income")
df["income_category"] = income_category
print(df["income_category"])
print(df)


house_age_category = []
for i in df['housing_median_age']:
    if i<=10:
        house_age_category.append("New House")
    elif i>10 and i<=30:
        house_age_category.append("Middle aged House")
    else:
        house_age_category.append("Old House")
df["house_age_category"] = house_age_category
print(df["house_age_category"])
print(df)

df.to_csv("C:\\Users\\HP\\OneDrive\\Documents\\Desktop\\summer_term\\Project\\pro\\finalise.csv")

### Housing Price Distribution using Histogram  
sns.set(style = 'whitegrid')
plt.figure(figsize=(10,6))
sns.histplot(df['median_house_value'],bins=30,kde=True,color='skyblue',edgecolor='navy')
plt.title("Housing Price Distribution")
plt.xlabel("Median House Value($)")
plt.ylabel("Number of Houses")
plt.tight_layout()

### Average House Value by Housing Age Group using Bargraph
plt.figure(figsize=(10,6))
avg_house_price = df.groupby('house_age_category')["median_house_value"].mean().reset_index()
sns.barplot(avg_house_price,x='house_age_category',y='median_house_value',hue='house_age_category',palette="pastel",legend=True)
plt.title("Average House Value by Housing Age Group")
plt.xlabel("House Age Category")
plt.ylabel("Median House Value")
plt.grid(True)

### Relationship Between Median Income and House Value using lineplot
plt.figure(figsize=(10, 5))
df_line = df.copy()
df_line['income_bin'] = pd.cut(df_line['median_income'], bins=50)
binned = df_line.groupby('income_bin', observed=False)[['median_income', 'median_house_value']].mean().reset_index()
sns.lineplot(data=binned, x='median_income', y='median_house_value', color='mediumorchid', marker='o',markersize = 5, markerfacecolor='indigo', markeredgewidth=1, markeredgecolor='darkgreen', linewidth=3)
plt.title("Relationship Between Median Income and House Value", fontsize=14,fontweight='bold')
plt.xlabel("Median Income ($)")
plt.ylabel("Median House Value ($)")
plt.tight_layout()

### Median Income vs House Value (Including $500K Cap) using Scatterplot
df_line = df.copy()
df_line['income_bin'] = pd.cut(df_line['median_income'], bins=50)
binned_avg = df_line.groupby('income_bin', observed=False)[['median_income', 'median_house_value']].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_line.sample(1000), x='median_income', y='median_house_value', color='teal', alpha=0.3, label='Raw Data')
sns.lineplot(data=binned_avg, x='median_income', y='median_house_value', color='crimson', label='Average Trend')
plt.title("Median Income vs House Value (Including $500K Cap)")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Distribution of Income Groups using Piechart
income_counts = df['income_category'].value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(income_counts,labels=income_counts.index, autopct='%1.1f%%',startangle=140, colors=sns.color_palette('pastel'))
plt.title("Distribution of Income Groups", fontsize=14)
plt.tight_layout()
plt.show()

### Distribution of Housing Age Categories using Donut Chart
age_counts = df['house_age_category'].value_counts().reindex(['New House', 'Middle aged House', 'Old House']).dropna()
plt.figure(figsize=(6, 6))
plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90,colors=sns.color_palette('Set2'), wedgeprops={'width': 0.4})
plt.title("Distribution of Housing Age Categories", fontsize=14)
plt.tight_layout()
plt.show()

### Showing Correlation Using HeatMap
df['Rooms_per_Household'] = df['total_rooms'] / df['households']

# Feature list
cols = [
    'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income', 'area_density', 'Rooms_per_Household',
    'median_house_value'
]
# Filter available columns only (handles any typos or missing columns)
available_cols = [col for col in cols if col in df.columns]
# Drop NA rows in those columns
corr_df = df[available_cols].dropna()
# Compute correlation matrix
correlation = corr_df.corr()
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title("Correlation Matrix of Key Housing Features", fontsize=14)
plt.tight_layout()
plt.show()

### BoxPlots of Numerical Features to show outliers
exclude_cols = ['area_density', 'Rooms_per_Household']
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in exclude_cols]

plt.figure(figsize=(16, 12))
plt.suptitle("Box Plots of Numerical Features", fontsize=18)

for i, col in enumerate(numerical_cols):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"Column: {col}")
    print(f"  - Lower Bound: {lower_bound:.2f}")
    print(f"  - Upper Bound: {upper_bound:.2f}")
    print(f"  - Outliers: {len(outliers)} ({(len(outliers) / len(df)) * 100:.2f}%)\n")

    plt.subplot((len(numerical_cols) + 2) // 3, 3, i + 1)
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(col)
    plt.xlabel("")

plt.tight_layout()
plt.show()

# Model Training Using Machine Learning
housingData = pd.read_csv("C:\\Users\\rajat\\Downloads\\cleaned(1).csv")
print('Print first few rows of this data -\n')
print(housingData.head())

# Drop unnecessary columns
cols_to_drop = ['Unnamed: 19', '<1H Ocean', 'Inland', 'Island', 'Near Bay', 'Near Ocean']
housingData = housingData.drop(columns=[col for col in cols_to_drop if col in housingData.columns])

# Separate input (X) and output (y)
X = housingData.drop(['median_house_value'], axis=1)
y = housingData['median_house_value'].values

# Handle missing values
# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute missing values in numeric columns
imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

# One-hot encode categorical columns
X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)

# Combine numeric and encoded categorical data
X_processed = pd.concat([X_numeric, X_categorical], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=0)

# Standardize data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Task1: Perform Linear Regression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
predictionLinear = linearRegression.predict(X_test)
mseLinear = mean_squared_error(y_test, predictionLinear)
print('Root mean squared error (RMSE) from Linear Regression =', np.sqrt(mseLinear))

# Task2: Perform Decision Tree Regression'''
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, y_train)
predictionDT = DTregressor.predict(X_test)
mseDT = mean_squared_error(y_test, predictionDT)
print('Root mean squared error from Decision Tree Regression =', np.sqrt(mseDT))


# Task3: Perform Random Forest Regression'''
RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, y_train)
predictionRF = RFregressor.predict(X_test)
mseRF = mean_squared_error(y_test, predictionRF)
print('Root mean squared error from Random Forest Regression =', np.sqrt(mseRF))

#Task4: Bonus exercise: Linear Regression with one feature
# Find index of 'median_income' in processed columns
feature_names = X_processed.columns.tolist()
median_income_index = feature_names.index('median_income')

# Extract only median_income from scaled X_train/X_test
X_train_median_income = X_train[:, [median_income_index]]
X_test_median_income = X_test[:, [median_income_index]]

# Train and predict
linearRegression2 = LinearRegression()
linearRegression2.fit(X_train_median_income, y_train)
predictionLinear2 = linearRegression2.predict(X_test_median_income)

# Visualize Training set
plt.scatter(X_train_median_income, y_train, color='green')
plt.plot(X_train_median_income, linearRegression2.predict(X_train_median_income), color='red')
plt.title('Training - median_income vs median_house_value')
plt.xlabel('median_income (standardized)')
plt.ylabel('median_house_value (standardized)')
plt.show()

# Visualize Testing set
plt.scatter(X_test_median_income, y_test, color='blue')
plt.plot(X_train_median_income, linearRegression2.predict(X_train_median_income), color='red')
plt.title('Testing - median_income vs median_house_value')
plt.xlabel('median_income (standardized)')
plt.ylabel('median_house_value (standardized)')
plt.show()

# Plotting actual vs predicted values to visually assess model accuracy for Linear Regression
plt.figure(figsize=(6,4))
plt.scatter(y_test, predictionLinear, color='purple', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Plotting actual vs predicted values to visually assess model accuracy for Decision Tree Regression
plt.figure(figsize=(6,4))
plt.scatter(y_test, predictionDT, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Decision Tree: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Plotting actual vs predicted values to visually assess model accuracy for Random Forest Regression
plt.figure(figsize=(6,4))
plt.scatter(y_test, predictionRF, color='orange', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Visualizing which features are most influential in predicting house prices using Random Forest
importances = RFregressor.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_processed.columns

plt.figure(figsize=(10,6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(X_processed.shape[1]), importances[indices], align='center')
plt.xticks(range(X_processed.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Plotting house price distribution across California based on longitude and latitude (hotspot map)
# Visualizing price by geo co-ordinates
plt.figure(figsize=(8,6))
plt.scatter(housingData['longitude'], housingData['latitude'],c=housingData['median_house_value'], cmap='viridis', s=10)
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California Housing Prices by Location')
plt.show()

# Displaying correlation between numeric features to understand linear relationships in the dataset
plt.figure(figsize=(10,6))
numeric_data = housingData.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
