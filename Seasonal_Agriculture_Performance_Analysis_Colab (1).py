# ============================================================
# MAJOR PROJECT: SEASONAL AGRICULTURE PERFORMANCE ANALYSIS
# ============================================================

# ---------------- CELL 1: IMPORT LIBRARIES ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

print("Libraries imported successfully.")


# ---------------- CELL 2: UPLOAD DATASET ----------------
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)

print("Dataset loaded successfully.")
print("File:", file_name)
print("Shape:", df.shape)


# ---------------- CELL 3: BASIC INFORMATION ----------------
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(i, "-", col)

print("\nFirst 5 rows:")
display(df.head())


# ---------------- CELL 4: DATA TYPES ----------------
print("Data Types:")
display(df.dtypes.to_frame("Data Type"))

print("\nDataset Information:")
df.info()


# ---------------- CELL 5: MISSING VALUES ----------------
missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing = missing[missing["Missing Values"] > 0]

print("Columns containing missing values:")
display(missing.sort_values("Missing Values", ascending=False))


# ---------------- CELL 6: DUPLICATES ----------------
print("Duplicate rows:", df.duplicated().sum())


# ---------------- CELL 7: UNIQUE CATEGORIES ----------------
categorical_columns = [
    "State",
    "District",
    "Crop",
    "Season",
    "Irrigation_Method"
]

for col in categorical_columns:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())


# ---------------- CELL 8: DATA CLEANING ----------------
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates().reset_index(drop=True)

print("Data cleaning completed.")
print("Remaining missing values:", df.isnull().sum().sum())
print("Final shape:", df.shape)


# ---------------- CELL 9: DESCRIPTIVE STATISTICS ----------------
print("Numerical Descriptive Statistics:")
display(df.describe().T)


# ---------------- CELL 10: SEASON DISTRIBUTION ----------------
season_count = df["Season"].value_counts()

print("Records by Season:")
display(season_count.to_frame("Number of Records"))

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Season", order=season_count.index)
plt.title("Distribution of Agricultural Records Across Seasons")
plt.xlabel("Season")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()


# ---------------- CELL 11: SEASONAL YIELD ----------------
season_yield = (
    df.groupby("Season")["Yield_Tonnes_Ha"]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
)

print("Yield Performance by Season:")
display(season_yield)

plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="Season", y="Yield_Tonnes_Ha")
plt.title("Yield Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("Yield (Tonnes/Ha)")
plt.tight_layout()
plt.show()


# ---------------- CELL 12: AVERAGE YIELD ----------------
yield_mean = (
    df.groupby("Season")["Yield_Tonnes_Ha"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))
sns.barplot(x=yield_mean.index, y=yield_mean.values)
plt.title("Average Agricultural Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")

for i, value in enumerate(yield_mean.values):
    plt.text(i, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# ---------------- CELL 13: PRODUCTION ----------------
season_production = (
    df.groupby("Season")["Production_Tonnes"]
    .agg(["mean", "median", "std", "sum"])
    .round(2)
)

print("Production Performance by Season:")
display(season_production)

production_mean = (
    df.groupby("Season")["Production_Tonnes"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))
sns.barplot(x=production_mean.index, y=production_mean.values)
plt.title("Average Production by Season")
plt.xlabel("Season")
plt.ylabel("Production (Tonnes)")
plt.tight_layout()
plt.show()


# ---------------- CELL 14: ECONOMIC PERFORMANCE ----------------
economic_season = (
    df.groupby("Season")[
        ["Total_Cost_INR", "Revenue_INR", "Profit_INR"]
    ].mean().round(2)
)

print("Average Economic Performance by Season:")
display(economic_season)

profit_mean = (
    df.groupby("Season")["Profit_INR"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9,5))
sns.barplot(x=profit_mean.index, y=profit_mean.values)
plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")
plt.axhline(0, linewidth=1)
plt.tight_layout()
plt.show()


# ---------------- CELL 15: PROFIT DISTRIBUTION ----------------
plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="Season", y="Profit_INR")
plt.title("Profit Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("Profit (INR)")
plt.tight_layout()
plt.show()


# ---------------- CELL 16: ENVIRONMENTAL ANALYSIS ----------------
environmental_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_Moisture_pct"
]

environmental_columns = [
    c for c in environmental_columns if c in df.columns
]

environmental_summary = (
    df.groupby("Season")[environmental_columns]
    .mean().round(2)
)

print("Average Environmental Conditions by Season:")
display(environmental_summary)


# ---------------- CELL 17: RAINFALL ----------------
if "Rainfall_mm" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Rainfall_mm")
    plt.title("Rainfall Variation Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Rainfall (mm)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 18: TEMPERATURE ----------------
if "Avg_Temperature_C" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Avg_Temperature_C")
    plt.title("Average Temperature Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 19: SOIL MOISTURE ----------------
if "Soil_Moisture_pct" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Soil_Moisture_pct")
    plt.title("Soil Moisture Variation Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Soil Moisture (%)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 20: RESOURCE USAGE ----------------
resource_columns = [
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Water_Used_m3"
]

resource_columns = [c for c in resource_columns if c in df.columns]

resource_summary = (
    df.groupby("Season")[resource_columns]
    .mean().round(2)
)

print("Average Resource Usage by Season:")
display(resource_summary)


# ---------------- CELL 21: FERTILIZER ----------------
if "Fertilizer_kg_ha" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Fertilizer_kg_ha")
    plt.title("Fertilizer Usage Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Fertilizer (kg/ha)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 22: WATER USAGE ----------------
if "Water_Used_m3" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Water_Used_m3")
    plt.title("Water Usage Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Water Used (m³)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 23: WATER EFFICIENCY ----------------
if "Water_Efficiency_t_per_1000m3" in df.columns:
    water_efficiency = (
        df.groupby("Season")["Water_Efficiency_t_per_1000m3"]
        .agg(["mean", "median", "std"]).round(3)
    )

    print("Water Efficiency by Season:")
    display(water_efficiency)

    plt.figure(figsize=(8,5))
    sns.barplot(
        data=df,
        x="Season",
        y="Water_Efficiency_t_per_1000m3"
    )
    plt.title("Average Water Efficiency by Season")
    plt.xlabel("Season")
    plt.ylabel("Tonnes per 1000 m³")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 24: DISEASE / PEST RISK ----------------
if "Disease_Pest_Risk_pct" in df.columns:
    risk_summary = (
        df.groupby("Season")["Disease_Pest_Risk_pct"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(2)
    )

    print("Disease/Pest Risk by Season:")
    display(risk_summary)

    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="Season", y="Disease_Pest_Risk_pct")
    plt.title("Disease and Pest Risk Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Disease/Pest Risk (%)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 25: CROP × SEASON YIELD ----------------
crop_season_yield = pd.pivot_table(
    df,
    values="Yield_Tonnes_Ha",
    index="Crop",
    columns="Season",
    aggfunc="mean"
)

print("Average Yield by Crop and Season:")
display(crop_season_yield.round(2))

plt.figure(figsize=(12,7))
sns.heatmap(crop_season_yield, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Average Crop Yield Across Seasons")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()


# ---------------- CELL 26: CROP × SEASON PROFIT ----------------
crop_profit = pd.pivot_table(
    df,
    values="Profit_INR",
    index="Crop",
    columns="Season",
    aggfunc="mean"
)

print("Average Profit by Crop and Season:")
display(crop_profit.round(0))

plt.figure(figsize=(12,7))
sns.heatmap(crop_profit, annot=True, fmt=".0f", cmap="RdYlGn", center=0)
plt.title("Average Crop Profit Across Seasons")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()


# ---------------- CELL 27: IRRIGATION METHOD ----------------
if "Irrigation_Method" in df.columns:
    irrigation_season = pd.crosstab(
        df["Season"], df["Irrigation_Method"]
    )

    print("Irrigation Method Distribution:")
    display(irrigation_season)

    irrigation_percentage = pd.crosstab(
        df["Season"],
        df["Irrigation_Method"],
        normalize="index"
    ) * 100

    print("Irrigation Method Percentage:")
    display(irrigation_percentage.round(2))

    irrigation_percentage.plot(
        kind="bar",
        stacked=True,
        figsize=(10,6)
    )

    plt.title("Irrigation Methods Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Percentage of Farms (%)")
    plt.legend(title="Irrigation Method")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ---------------- CELL 28: SEED QUALITY ----------------
if "Seed_Quality_Score" in df.columns:
    seed_quality = (
        df.groupby("Season")["Seed_Quality_Score"]
        .agg(["mean", "median", "std"]).round(3)
    )

    print("Seed Quality by Season:")
    display(seed_quality)


# ---------------- CELL 29: MARKET PRICE ----------------
if "Market_Price_INR_Tonne" in df.columns:
    market_price = (
        df.groupby("Season")["Market_Price_INR_Tonne"]
        .agg(["mean", "median", "std"]).round(2)
    )

    print("Market Price by Season:")
    display(market_price)

    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df,
        x="Season",
        y="Market_Price_INR_Tonne"
    )
    plt.title("Market Price Variation Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Market Price (INR/Tonne)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 30: STATE × SEASON YIELD ----------------
if "State" in df.columns:
    state_season_yield = pd.pivot_table(
        df,
        values="Yield_Tonnes_Ha",
        index="State",
        columns="Season",
        aggfunc="mean"
    )

    print("Average Yield by State and Season:")
    display(state_season_yield.round(2))

    plt.figure(figsize=(14,8))
    sns.heatmap(
        state_season_yield,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu"
    )
    plt.title("Seasonal Yield Variation Across States")
    plt.xlabel("Season")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 31: CORRELATION ----------------
correlation_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Seed_Quality_Score",
    "Yield_Tonnes_Ha",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3",
    "Disease_Pest_Risk_pct",
    "Profit_INR"
]

correlation_columns = [c for c in correlation_columns if c in df.columns]

correlation_matrix = df[correlation_columns].corr()

print("Correlation Matrix:")
display(correlation_matrix.round(2))

plt.figure(figsize=(15,11))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Between Agricultural Variables")
plt.tight_layout()
plt.show()


# ---------------- CELL 32: CORRELATION WITH YIELD ----------------
yield_correlation = (
    correlation_matrix["Yield_Tonnes_Ha"]
    .sort_values(ascending=False)
)

print("Factors Correlated with Yield:")
display(yield_correlation.to_frame("Correlation with Yield"))


# ---------------- CELL 33: CORRELATION WITH PROFIT ----------------
profit_correlation = (
    correlation_matrix["Profit_INR"]
    .sort_values(ascending=False)
)

print("Factors Correlated with Profit:")
display(profit_correlation.to_frame("Correlation with Profit"))


# ---------------- CELL 34: ANOVA - YIELD ----------------
season_names = df["Season"].dropna().unique()

season_groups = [
    df.loc[df["Season"] == season, "Yield_Tonnes_Ha"]
    for season in season_names
]

anova_result = stats.f_oneway(*season_groups)

print("ONE-WAY ANOVA — Seasonal Yield")
print("--------------------------------")
print("F-statistic:", round(anova_result.statistic, 4))
print("p-value:", round(anova_result.pvalue, 6))

if anova_result.pvalue < 0.05:
    print("Conclusion: Seasonal yield differences are statistically significant.")
else:
    print("Conclusion: Seasonal yield differences are not statistically significant.")


# ---------------- CELL 35: ANOVA - PROFIT ----------------
profit_groups = [
    df.loc[df["Season"] == season, "Profit_INR"]
    for season in season_names
]

profit_anova = stats.f_oneway(*profit_groups)

print("ONE-WAY ANOVA — Seasonal Profit")
print("--------------------------------")
print("F-statistic:", round(profit_anova.statistic, 4))
print("p-value:", round(profit_anova.pvalue, 6))

if profit_anova.pvalue < 0.05:
    print("Conclusion: Seasonal profit differences are statistically significant.")
else:
    print("Conclusion: Seasonal profit differences are not statistically significant.")


# ---------------- CELL 36: ENVIRONMENTAL ANOVA ----------------
environmental_tests = [
    c for c in [
        "Rainfall_mm",
        "Avg_Temperature_C",
        "Humidity_pct",
        "Soil_Moisture_pct",
        "Sunlight_Hours_Day"
    ] if c in df.columns
]

anova_results = []

for variable in environmental_tests:

    groups = [
        df.loc[df["Season"] == season, variable]
        for season in season_names
    ]

    result = stats.f_oneway(*groups)

    anova_results.append({
        "Variable": variable,
        "F-statistic": result.statistic,
        "p-value": result.pvalue,
        "Significant": "Yes" if result.pvalue < 0.05 else "No"
    })

environmental_anova = pd.DataFrame(anova_results)

print("Environmental ANOVA Results:")
display(environmental_anova.round(5))


# ---------------- CELL 37: OUTLIER ANALYSIS ----------------
outlier_columns = [
    c for c in [
        "Yield_Tonnes_Ha",
        "Profit_INR",
        "Water_Used_m3",
        "Rainfall_mm",
        "Disease_Pest_Risk_pct"
    ] if c in df.columns
]

outlier_summary = []

for col in outlier_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_summary.append({
        "Variable": col,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Outlier Count": count,
        "Outlier %": (count / len(df)) * 100
    })

outlier_df = pd.DataFrame(outlier_summary)

print("Outlier Analysis:")
display(outlier_df.round(2))


# ---------------- CELL 38: BEST CROP BY SEASON ----------------
best_crop_each_season = (
    df.groupby(["Season", "Crop"])["Yield_Tonnes_Ha"]
    .mean()
    .reset_index()
)

best_crop_each_season = best_crop_each_season.loc[
    best_crop_each_season.groupby("Season")["Yield_Tonnes_Ha"].idxmax()
]

print("Best Yielding Crop in Each Season:")
display(
    best_crop_each_season
    .sort_values("Season")
    .reset_index(drop=True)
)


# ---------------- CELL 39: MOST PROFITABLE CROP ----------------
best_profit_crop = (
    df.groupby(["Season", "Crop"])["Profit_INR"]
    .mean()
    .reset_index()
)

best_profit_crop = best_profit_crop.loc[
    best_profit_crop.groupby("Season")["Profit_INR"].idxmax()
]

print("Most Profitable Crop in Each Season:")
display(
    best_profit_crop
    .sort_values("Season")
    .reset_index(drop=True)
)


# ---------------- CELL 40: MOST WATER-EFFICIENT CROP ----------------
if "Water_Efficiency_t_per_1000m3" in df.columns:

    best_water_crop = (
        df.groupby(["Season", "Crop"])
        ["Water_Efficiency_t_per_1000m3"]
        .mean()
        .reset_index()
    )

    best_water_crop = best_water_crop.loc[
        best_water_crop.groupby("Season")
        ["Water_Efficiency_t_per_1000m3"]
        .idxmax()
    ]

    print("Most Water-Efficient Crop in Each Season:")
    display(
        best_water_crop
        .sort_values("Season")
        .reset_index(drop=True)
    )


# ---------------- CELL 41: HIGH-RISK CONDITIONS ----------------
if "Disease_Pest_Risk_pct" in df.columns:

    risk_threshold = df["Disease_Pest_Risk_pct"].quantile(0.75)

    high_risk = df[
        df["Disease_Pest_Risk_pct"] >= risk_threshold
    ]

    print(
        f"High-risk threshold (75th percentile): "
        f"{risk_threshold:.2f}%"
    )

    print("High-risk observations:", len(high_risk))

    high_risk_season = (
        high_risk["Season"]
        .value_counts()
        .to_frame("High Risk Observations")
    )

    display(high_risk_season)


# ---------------- CELL 42: YIELD VS RAINFALL ----------------
if "Rainfall_mm" in df.columns:
    plt.figure(figsize=(9,6))
    sns.scatterplot(
        data=df,
        x="Rainfall_mm",
        y="Yield_Tonnes_Ha",
        hue="Season",
        alpha=0.6
    )
    plt.title("Relationship Between Rainfall and Agricultural Yield")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Yield (Tonnes/Ha)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 43: YIELD VS SOIL MOISTURE ----------------
if "Soil_Moisture_pct" in df.columns:
    plt.figure(figsize=(9,6))
    sns.scatterplot(
        data=df,
        x="Soil_Moisture_pct",
        y="Yield_Tonnes_Ha",
        hue="Season",
        alpha=0.6
    )
    plt.title("Relationship Between Soil Moisture and Yield")
    plt.xlabel("Soil Moisture (%)")
    plt.ylabel("Yield (Tonnes/Ha)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 44: YIELD VS TEMPERATURE ----------------
if "Avg_Temperature_C" in df.columns:
    plt.figure(figsize=(9,6))
    sns.scatterplot(
        data=df,
        x="Avg_Temperature_C",
        y="Yield_Tonnes_Ha",
        hue="Season",
        alpha=0.6
    )
    plt.title("Relationship Between Temperature and Yield")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Yield (Tonnes/Ha)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 45: YIELD VS WATER ----------------
if "Water_Used_m3" in df.columns:
    plt.figure(figsize=(9,6))
    sns.scatterplot(
        data=df,
        x="Water_Used_m3",
        y="Yield_Tonnes_Ha",
        hue="Season",
        alpha=0.6
    )
    plt.title("Relationship Between Water Usage and Yield")
    plt.xlabel("Water Used (m³)")
    plt.ylabel("Yield (Tonnes/Ha)")
    plt.tight_layout()
    plt.show()


# ---------------- CELL 46: PROFIT VS YIELD ----------------
plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df,
    x="Yield_Tonnes_Ha",
    y="Profit_INR",
    hue="Season",
    alpha=0.6
)
plt.title("Relationship Between Yield and Profit")
plt.xlabel("Yield (Tonnes/Ha)")
plt.ylabel("Profit (INR)")
plt.tight_layout()
plt.show()


# ---------------- CELL 47: SEASONAL PERFORMANCE INDEX ----------------
index_columns = [
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Water_Efficiency_t_per_1000m3"
]

index_columns = [c for c in index_columns if c in df.columns]

season_index = df.groupby("Season")[index_columns].mean()

for col in index_columns:

    min_val = season_index[col].min()
    max_val = season_index[col].max()

    if max_val != min_val:
        season_index[col + "_Normalized"] = (
            (season_index[col] - min_val) /
            (max_val - min_val)
        )
    else:
        season_index[col + "_Normalized"] = 1

# Weights
if all(
    c in season_index.columns for c in [
        "Yield_Tonnes_Ha_Normalized",
        "Profit_INR_Normalized",
        "Water_Efficiency_t_per_1000m3_Normalized"
    ]
):
    season_index["Performance_Index"] = (
        season_index["Yield_Tonnes_Ha_Normalized"] * 0.40 +
        season_index["Profit_INR_Normalized"] * 0.40 +
        season_index["Water_Efficiency_t_per_1000m3_Normalized"] * 0.20
    )
else:
    normalized_cols = [
        c for c in season_index.columns
        if c.endswith("_Normalized")
    ]
    season_index["Performance_Index"] = (
        season_index[normalized_cols].mean(axis=1)
    )

season_index = season_index.sort_values(
    "Performance_Index",
    ascending=False
)

print("Seasonal Performance Index:")
display(season_index.round(3))


# ---------------- CELL 48: FINAL SEASON RANKING ----------------
final_ranking = season_index[["Performance_Index"]].copy()

final_ranking["Rank"] = (
    final_ranking["Performance_Index"]
    .rank(ascending=False)
    .astype(int)
)

final_ranking = final_ranking.sort_values("Rank")

print("FINAL SEASON RANKING:")
display(final_ranking)


# ---------------- CELL 49: KEY FINDINGS ----------------
best_yield_season = (
    df.groupby("Season")["Yield_Tonnes_Ha"]
    .mean().idxmax()
)

best_yield_value = (
    df.groupby("Season")["Yield_Tonnes_Ha"]
    .mean().max()
)

best_profit_season = (
    df.groupby("Season")["Profit_INR"]
    .mean().idxmax()
)

best_profit_value = (
    df.groupby("Season")["Profit_INR"]
    .mean().max()
)

if "Water_Efficiency_t_per_1000m3" in df.columns:
    best_water_season = (
        df.groupby("Season")
        ["Water_Efficiency_t_per_1000m3"]
        .mean().idxmax()
    )

    best_water_value = (
        df.groupby("Season")
        ["Water_Efficiency_t_per_1000m3"]
        .mean().max()
    )

if "Disease_Pest_Risk_pct" in df.columns:
    highest_risk_season = (
        df.groupby("Season")["Disease_Pest_Risk_pct"]
        .mean().idxmax()
    )

    highest_risk_value = (
        df.groupby("Season")["Disease_Pest_Risk_pct"]
        .mean().max()
    )

print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print(
    f"\n1. Highest average yield: {best_yield_season} "
    f"({best_yield_value:.2f} Tonnes/Ha)"
)

print(
    f"\n2. Highest average profit: {best_profit_season} "
    f"(₹{best_profit_value:,.2f})"
)

if "Water_Efficiency_t_per_1000m3" in df.columns:
    print(
        f"\n3. Highest water efficiency: {best_water_season} "
        f"({best_water_value:.2f} tonnes/1000 m³)"
    )

if "Disease_Pest_Risk_pct" in df.columns:
    print(
        f"\n4. Highest average disease/pest risk: "
        f"{highest_risk_season} ({highest_risk_value:.2f}%)"
    )

print("\n5. Seasonal Performance Ranking:")
display(final_ranking)


# ---------------- CELL 50: FINAL SUMMARY ----------------
aggregation = {
    "Avg_Yield": ("Yield_Tonnes_Ha", "mean"),
    "Avg_Production": ("Production_Tonnes", "mean"),
    "Avg_Cost": ("Total_Cost_INR", "mean"),
    "Avg_Revenue": ("Revenue_INR", "mean"),
    "Avg_Profit": ("Profit_INR", "mean")
}

if "Farm_ID" in df.columns:
    aggregation["Farms"] = ("Farm_ID", "count")
else:
    aggregation["Farms"] = ("Season", "count")

if "Water_Used_m3" in df.columns:
    aggregation["Avg_Water_Use"] = ("Water_Used_m3", "mean")

if "Water_Efficiency_t_per_1000m3" in df.columns:
    aggregation["Avg_Water_Efficiency"] = (
        "Water_Efficiency_t_per_1000m3", "mean"
    )

if "Disease_Pest_Risk_pct" in df.columns:
    aggregation["Avg_Disease_Risk"] = (
        "Disease_Pest_Risk_pct", "mean"
    )

final_summary = df.groupby("Season").agg(**aggregation).round(2)

print("FINAL SEASONAL AGRICULTURAL PERFORMANCE SUMMARY:")
display(final_summary)


# ---------------- CELL 51: SAVE RESULTS ----------------
df.to_csv(
    "cleaned_seasonal_agriculture_dataset.csv",
    index=False
)

season_performance.to_csv("season_performance.csv")
final_summary.to_csv("final_seasonal_summary.csv")
crop_season_yield.to_csv("crop_season_yield.csv")
state_season_yield.to_csv("state_season_yield.csv")
environmental_summary.to_csv("environmental_summary.csv")
resource_summary.to_csv("resource_summary.csv")
environmental_anova.to_csv("environmental_anova.csv")
outlier_df.to_csv("outlier_analysis.csv")

print("All analysis files have been saved.")


# ---------------- CELL 52: DOWNLOAD FINAL SUMMARY ----------------
files.download("final_seasonal_summary.csv")

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY.")
