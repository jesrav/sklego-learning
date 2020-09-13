import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklego.preprocessing import ColumnSelector
from sklego.meta import GroupedPredictor
from sklego.linear_model import LowessRegression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def load_data():
    return pd.read_csv("data\daily-bike-share.csv")


def create_features(df):
    return df.copy().assign(time_index=df.index)


df = load_data().pipe(create_features)

##############################################################
# Grouped predictor
# Create seperate model for seperate groups.
##############################################################
features = ["day", "mnth", "year", "season", "holiday", "weekday"]
group_features = ["season"]

regressor = GroupedPredictor(LinearRegression(), groups=group_features)

pipeline = Pipeline(
    [("grap_cols", ColumnSelector(features)), ("regression", regressor)]
)
pipeline.fit(df, df.rentals)

y_hat = pipeline.predict(df)
y_hat_shuffled_features = pipeline.predict(
    df[["holiday", "weekday", "day", "mnth", "year", "season"]]
)

assert all(
    y_hat == y_hat_shuffled_features
), "Shuffling feature order leads to different results."

# Notice that not having a columns throws an error.
pipeline.predict(df.drop("year", axis=1))


##############################################################
# Lowess smoothing
##############################################################
lowess = LowessRegression(sigma=100, span=1)
lowess = lowess.fit(df["time_index"].values.reshape(-1, 1), df.rentals)
preds = lowess.predict(df["time_index"].values.reshape(-1, 1))

fig, ax = plt.subplots(nrows=1, ncols=1)
df.plot(kind="scatter", x="time_index", y="rentals", ax=ax)
plt.plot(df.time_index, preds, color="orange")
fig.savefig("rental_fig.png")

##############################################################
# Lowess smoothing
##############################################################
