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

RANDOM_STATE = 202605

# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")

# We load all transactions made in France between 2010 and 2024
trans = con.sql(
    """
        SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()

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

p99 = np.percentile(y, 99)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

for ax, (data, label) in zip(
    axes,
    [
        (y, "Y"),
        (y[y <= p], "Y filtered"),
        (np.log(y), "log(Y)"),
        (np.log(y[y <= p]), "log(Y) filtered"),
    ],
):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.5)

    # omeji x os na 99 % podatkov
    xmax = np.percentile(data, 99)
    ax.set_xlim(data.min(), xmax)

    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()







# %%
# ============================================
# STEP 2 — Train / test split, model fitting,
#
# ============================================

# YOUR CODE HERE

# %%
