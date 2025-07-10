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

