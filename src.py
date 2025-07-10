### Cleaning of dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading of dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Desktop\\summer_term\\Project\\pro\\housing.csv")

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

