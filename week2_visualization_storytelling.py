import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_cleaned.csv")

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["age_group"] = pd.cut(
    df["age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    include_lowest=True
)

# 1. Survival by class
s = df.groupby("pclass")["survived"].mean() * 100
s.plot(kind="bar", figsize=(7, 4.5))
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig("01_survival_by_class.png", dpi=180); plt.show()

# 2. Survival by sex and class
p = df.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean") * 100
p.plot(kind="bar", figsize=(7, 4.5))
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig("02_survival_sex_class.png", dpi=180); plt.show()

# 3. Survival by age group
a = df.groupby("age_group", observed=False)["survived"].mean() * 100
a.plot(kind="line", marker="o", figsize=(7, 4.5))
plt.title("Survival Rate Across Age Groups")
plt.xlabel("Age Group"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig("03_survival_by_age_group.png", dpi=180); plt.show()

# 4. Fare and age by survival
for outcome, group in df.groupby("survived"):
    plt.scatter(group["age"], group["fare"], alpha=0.35,
                label="Survived" if outcome == 1 else "Did not survive")
plt.title("Fare and Age Relationship by Survival Outcome")
plt.xlabel("Age"); plt.ylabel("Fare"); plt.legend()
plt.tight_layout(); plt.savefig("04_fare_age_survival.png", dpi=180); plt.show()

# 5. Survival by family size
f = df.groupby("family_size")["survived"].mean().mul(100)
f[f.index <= 8].plot(kind="bar", figsize=(7, 4.5))
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig("05_survival_family_size.png", dpi=180); plt.show()

# 6. Correlation matrix
cols = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]
c = df[cols].corr()
plt.figure(figsize=(7, 5))
plt.imshow(c.values, aspect="auto")
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)
plt.title("Correlation Matrix")
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(); plt.tight_layout()
plt.savefig("06_correlation_matrix.png", dpi=180); plt.show()
