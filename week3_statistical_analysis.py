import pandas as pd
from scipy import stats

df = pd.read_csv("titanic_cleaned.csv")

# H1: Sex and survival are associated.
contingency = pd.crosstab(df["sex"], df["survived"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print("Chi-square statistic:", chi2)
print("Chi-square p-value:", p_value)

# H2: Mean fare differs between survivors and non-survivors.
survived_fare = df.loc[df["survived"] == 1, "fare"].dropna()
not_survived_fare = df.loc[df["survived"] == 0, "fare"].dropna()
t_stat, p_value = stats.ttest_ind(
    survived_fare, not_survived_fare, equal_var=False
)
print("Welch t statistic:", t_stat)
print("Welch t-test p-value:", p_value)

# H3: Mean fare differs among passenger classes.
fare_groups = [
    group["fare"].dropna()
    for _, group in df.groupby("pclass")
]
f_stat, p_value = stats.f_oneway(*fare_groups)
print("ANOVA F statistic:", f_stat)
print("ANOVA p-value:", p_value)

# 95% CI for the difference in mean fare
print("Mean fare - survivors:", survived_fare.mean())
print("Mean fare - non-survivors:", not_survived_fare.mean())
