import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest

# Load CSV file
df = pd.read_csv("5_6305544828465714708.csv")
print("CSV loaded successfully.")

# Cleaning data
numeric_cols = df.select_dtypes(include=['float64','int64']).columns

df[numeric_cols] = df[numeric_cols].replace(0, np.nan)

for column in numeric_cols:
    df.fillna({column: df[column].mean()}, inplace=True)
df.dropna(subset=['ANNUAL'],inplace=True)

print("Dataset cleaned.")

# First 18 rows
print("First 18 rows: \n",df.head(18))
# Overview
print("Total no. of rows and columns: \n",df.shape)
print("Quick summary: \n",df.info)
print("Name of columns: \n",df.columns)
print("Data type of each column: \n",df.dtypes)
# Summary of numerical columns
print("Summary of columns: \n",df.describe())
# Unique values
print("Unique subdivisions: \n",df['SUBDIVISION'].unique())
print("Year range: \n",df['YEAR'].min(),df['YEAR'].max())
print("Frequency of records per subdivision: \n",df['SUBDIVISION'].value_counts())

# 1.Predict ANNUAL rainfall trends by subdivision
selected_subdivision = ['Andaman & Nicobar Islands','Arunachal Pradesh','Punjab','Kerala','Uttarakhand']
df_filtered = df[df['SUBDIVISION'].isin(selected_subdivision)]

plt.figure(figsize=(12,6))
sns.lineplot(data=df_filtered, x="YEAR", y="ANNUAL", hue="SUBDIVISION", marker='o')
plt.title("Annual Rainfall Trends (Selected Subdivision)")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.legend(title="Subdivision")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2.Analyze Seasonal Rainfall Variability
seasonal_avg = df.groupby('SUBDIVISION')[['JF','MAM','JJAS','OND']].mean().reset_index()
seasonal_melted = seasonal_avg.melt(id_vars='SUBDIVISION',var_name='Season',value_name='Rainfall')
selected_subdivisions=['Uttarakhand','Kerala','Arunachal Pradesh']
seasonal_melted = seasonal_melted[seasonal_melted['SUBDIVISION'].isin(selected_subdivisions)]

plt.figure(figsize=(12,6))
sns.barplot(data=seasonal_melted, x='SUBDIVISION', y='Rainfall', hue='Season')
plt.title("Average Seasonal Rainfall by Subdivision")
plt.ylabel("Rainfall (mm)")
plt.ylabel("Subdivision")
plt.legend(title='Season')
plt.tight_layout()
plt.show()

# 3.Compare Rainfall Trends Between Coastal and Inland Subdivision
coastal = ['Kerala','Tamil Nadu','Odisha','Konkan & Goa','Coastal Karnataka','Andhra Pradesh Coastal']
inland = ['Vidrabha','Punjab','Haryana','West Madhya Pradesh','Bihar','Uttar Pradesh East']

df['Region'] = df['SUBDIVISION'].apply(lambda x: 'Coastal' if x in coastal else 'Inland' if x in inland else 'Other')
df_filtered = df[df['Region'] != 'Other']

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_filtered, x='YEAR',y='ANNUAL', hue='Region', alpha=0.6)
plt.title("Annual Rainfall Trends: Coastal vs Inland Subdivision")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4.Annual Rainfall Distribution in Major Food-Producing Regions
selected_subs = ['Punjab','Uttarakhand','Bihar','East Uttar Pradesh']
df_food = df[df['SUBDIVISION'].isin(selected_subs)]

plt.figure(figsize=(10,6))
sns.boxplot(data=df_food, x='SUBDIVISION', y='ANNUAL', palette='Set2', hue='SUBDIVISION')
plt.legend([],[], frameon=False)
plt.title("Annual Rainfall Distribution in Major Food-Producing Regions")
plt.xlabel("Subdivision")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5.Detect Yearly Rainfall Anomalies across subdivisions
selected_subs = ['Kerala', 'Punjab', 'Tamil Nadu', 'Bihar', 'Gujarat Region', 'East Rajasthan']
df_filtered = df[df['SUBDIVISION'].isin(selected_subs) & (df['YEAR'] >= 1990)]
pivot_df = df_filtered.pivot(index='YEAR', columns='SUBDIVISION', values='ANNUAL')
normalized_df = pivot_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

plt.figure(figsize=(12, 6))
sns.heatmap(normalized_df.T, cmap='RdBu_r', center=0, annot=True, fmt=".2f",
            cbar_kws={'label': 'Z-score (Rainfall Anomaly)'}, linewidths=0.3)
plt.title("Rainfall Anomalies (1990–2017) in Selected Subdivisions")
plt.xlabel("Year")
plt.ylabel("Subdivision")
plt.tight_layout()
plt.show()

# 6.Is the average annual rainfall in Kerala significantly different from the national average
kerala_rainfall = df[df["SUBDIVISION"] == "Kerala"]["ANNUAL"]
all_rainfall = df["ANNUAL"]
z_stat, p_value = ztest(kerala_rainfall, value=all_rainfall.mean())
print(f"Z-statistic: {z_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Significant difference in rainfall (reject null hypothesis).")
else:
    print("Result: No significant difference in rainfall (fail to reject null).")