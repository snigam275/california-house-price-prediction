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

### 
