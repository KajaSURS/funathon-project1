# %%
# ============================================
# STEP 1 — Preprocessing
# ============================================

# YOUR CODE HERE

import duckdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_STATE = 202605

# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")

# We load all transactions made in France between 2010 and 2024
trans = con.sql(
    """
        SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()

# EDA part

trans.shape # (nb_rows, nb_columns)
trans.dtypes # type of each columns
trans.columns # list of columns
trans.index # index
trans.info() # full summary: types + non-null values + memory
trans.describe() # statistics (count, mean, std, min, max, quartiles)
trans.head(n=5) # first n rows
trans.isnull().sum() # nb of NaN per column
trans.notnull().all() # column with no NaN?

trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]

trans["price_sqm"] = trans["price"] / trans["farea"]


y = trans["price_sqm"]
p = np.percentile(y, 98.5)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

for ax, (data, label) in zip(axes, [(y, "Y"), (y[y <= p], "Y filtered"), (np.log(y), "log(Y)"), (np.log(y[y <= p]), "log(Y) filtered")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# the distribution of the target y for y below 2000€ per square meter and below 500 € per sqm.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (data, label) in zip(axes, [(y[y <= 2000], "Y below 2000€ per sqm"), (y[y <= 500], "Y below 500€ per sqm")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.95)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# OUTLIERS REMOVAL
# HINT "Interquartile Range method can keep all observations with target value between 
# D1-1.5*(D9-D1) and D9+1.5*(D9-D1) with D1 and D9 the first and ninth deciles.""
# KajaSURS: In the hint is a decile based difference (IDR interdecile range), not interquartile range. 
# Would apply a model to detect extreme and inconsistent values.

n0 = trans.shape[0]
print(f"{n0} rows before filtering")

# Apply some deterministic threshold on the dataframe
trans = trans[(trans["price_sqm"] < 200000) & (trans["price_sqm"] > 100)]

print(f"{trans.shape[0]} rows after deterministic filtering")

# Apply IQR methods for the outlier removal - interdecile range 
def outlier_transform(y, lower=0.1, upper=0.9):
    """
    Transform Y target to log(Y) and remove outliers with IQR method

    Args :
        y : target
        lower: lower quantile for the IQR
        upper: upper quantile for the IQR
    """
    Q_lower = np.quantile(y, lower)
    Q_upper = np.quantile(y, upper)
    IQR = Q_upper - Q_lower

    mask = (y >= Q_lower - 1.5 * IQR) & (y <= Q_upper + 1.5 * IQR)
    return mask

mask = outlier_transform(trans["price_sqm"])
trans = trans[mask].reset_index(drop=True)

n1 = trans.shape[0]

print(f"{n1} rows after deterministic and statistic filtering")
print(f'Applying these filters methods has dropped about {((n0 - n1)/n0)*100:.2f} % of the transactions.')

### DROP MISSING

trans = trans.dropna(subset = "price_sqm")

df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])

# Printing all rows containing at least one NA
print(df[df.isna().any(axis=1)])

# Filtering NA values
df = df.dropna()

df["prop_type"] = pd.Categorical(
    df["prop_type"].astype(str),
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

counts = df.value_counts("prop_year_harm").reset_index()
counts[counts["prop_year_harm"] < 1850].describe() # there more than 500 different years of construction, going from 13th century to now. Maybe we can bundle together years before 1850 and group them by decade

counts_10 = ((df["prop_year_harm"] // 10)*10).value_counts().reset_index()  # 82 modalities
counts_10[counts_10["prop_year_harm"] < 1850].describe()  # years before 1850 represent 64 modalities with maximal class of about two thousands operations - ok
counts_10[counts_10["prop_year_harm"] < 1850]["count"].sum()

# Replacing year of construction by decade and merging together all years before 1850
df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])

# %%
# ============================================
# STEP 2 — Train / test split, model fitting,
#
# ============================================

# YOUR CODE HERE

# Split features / target
X = df.drop(columns=["price_sqm"])  # X must contain only the features we'll learn from
y = df["price_sqm"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)



# %%
