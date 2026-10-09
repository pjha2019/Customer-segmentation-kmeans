# Customer Segmentation using K-Means Clustering

Segments customers into groups based on annual income and spending score so marketing can target each group differently.

## Tools
Python (Pandas, NumPy, Matplotlib, scikit-learn), Power BI

## Steps
1. Load and check the data (5 columns: CustomerID, Gender, Age, Annual_Income_k, Spending_Score)
2. Basic cleaning (missing values, duplicates) and a scatter plot
3. Scale features with StandardScaler
4. Choose number of clusters using the elbow method (inertia)
5. Fit K-Means and label each customer with a cluster
6. Profile each segment (size, average income, spending, age)
7. Export `customers_segmented.csv` and build a Power BI dashboard from it

## Data
This is a small sample dataset. The script generates a sample `customers.csv` if one is not present; replace it with your own file with the same column names to re-run.

## Run
```
pip install -r requirements.txt
python segmentation.py
```

## Power BI dashboard
Load `customers_segmented.csv` and `segment_summary.csv`. Suggested visuals: scatter plot (income vs spending, coloured by Cluster), bar chart of customers per cluster, table of segment averages, and a Gender slicer.

## Segment ideas (fill in after you view your results)
- High income, high spending: loyalty / premium offers
- High income, low spending: re-engagement campaigns
- Low income, high spending: value offers
- Low income, low spending: low-cost communication
