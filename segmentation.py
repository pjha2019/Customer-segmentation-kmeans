"""
Customer Segmentation using K-Means Clustering
Basic project: segment customers by annual income and spending score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------- 1. Load data ----------
# Sample dataset with 5 columns: CustomerID, Gender, Age, Annual Income, Spending Score.
# If customers.csv doesn't exist, a sample file is created so the project runs end to end.
try:
    df = pd.read_csv("customers.csv")
except FileNotFoundError:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 70, n),
        "Annual_Income_k": np.clip(rng.normal(60, 25, n), 15, 140).round(0),
        "Spending_Score": np.clip(rng.normal(50, 25, n), 1, 99).round(0),
    })
    df.to_csv("customers.csv", index=False)

print(df.head())
print(df.info())
print(df.describe())

# ---------- 2. Basic cleaning / EDA ----------
print("Missing values:\n", df.isnull().sum())
df = df.drop_duplicates()

plt.figure(figsize=(6, 4))
plt.scatter(df["Annual_Income_k"], df["Spending_Score"], alpha=0.7)
plt.xlabel("Annual Income (k)")
plt.ylabel("Spending Score")
plt.title("Income vs Spending Score")
plt.savefig("eda_scatter.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 3. Prepare features ----------
X = df[["Annual_Income_k", "Spending_Score"]]
X_scaled = StandardScaler().fit_transform(X)

# ---------- 4. Choose k with the elbow method ----------
inertia = []
k_range = range(1, 11)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(k_range), inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_plot.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 5. Fit final model ----------
# Choose k from the elbow plot (5 is typical for this kind of data).
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------- 6. Visualise clusters ----------
plt.figure(figsize=(6, 4))
plt.scatter(df["Annual_Income_k"], df["Spending_Score"], c=df["Cluster"], cmap="viridis", alpha=0.8)
plt.xlabel("Annual Income (k)")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.savefig("clusters.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 7. Profile each segment ----------
summary = df.groupby("Cluster").agg(
    customers=("CustomerID", "count"),
    avg_income=("Annual_Income_k", "mean"),
    avg_spending=("Spending_Score", "mean"),
    avg_age=("Age", "mean"),
).round(1)
print(summary)

# Save output for Power BI
df.to_csv("customers_segmented.csv", index=False)
summary.to_csv("segment_summary.csv")
print("Saved customers_segmented.csv for the Power BI dashboard.")
