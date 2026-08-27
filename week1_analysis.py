"""
NSDC Week 1 - Data Acquisition, Cleaning and Exploratory Data Analysis
Dataset: Titanic Passenger Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")

# Inspect the dataset
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

print("\nOverall survival rate:", df["survived"].mean())

# Age distribution
plt.figure(figsize=(7, 4.5))
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("01_age_distribution.png", dpi=180)
plt.show()

# Survival count by sex
pd.crosstab(df["sex"], df["survived"]).plot(kind="bar", figsize=(7, 4.5))
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("02_survival_by_sex.png", dpi=180)
plt.show()

# Fare distribution by passenger class
df.boxplot(column="fare", by="pclass", figsize=(7, 4.5))
plt.title("Fare Distribution by Passenger Class")
plt.suptitle("")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("03_fare_by_class.png", dpi=180)
plt.show()

# Correlation matrix
numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation = df[numeric_columns].corr()

plt.figure(figsize=(7, 5))
plt.imshow(correlation.values, aspect="auto")
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
plt.yticks(range(len(numeric_columns)), numeric_columns)
plt.title("Correlation Matrix")

for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8)

plt.colorbar()
plt.tight_layout()
plt.savefig("04_correlation_matrix.png", dpi=180)
plt.show()

print("\nAnalysis completed successfully.")
