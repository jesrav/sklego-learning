import pandas as pd
from sklearn.pipeline import Pipeline
from sklego.preprocessing import ColumnSelector
from sklego.meta import GroupedPredictor
from sklego.linear_model import LowessRegression
from sklearn.linear_model import LinearRegression
from sklego.preprocessing import RepeatingBasisFunction
from matplotlib import pyplot as plt


def load_data():
    return pd.read_csv("data\daily-bike-share.csv")


def create_features(df):
    return df.copy().assign(time_index=df.index)


df = load_data().pipe(create_features)

##############################################################
# Grouped predictor
# Create separate model for separate groups.
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

# Plot predictions and actual time series
fig, ax = plt.subplots(nrows=1, ncols=1)
df.plot(kind="scatter", x="time_index", y="rentals", ax=ax)
plt.plot(df.time_index, preds, color="orange")
fig.savefig("rental_lowess.png")


##############################################################
# Using repeating basis functions for prepossessing time series
##############################################################

# Transform time feature and inspect rbf features.
rbf = RepeatingBasisFunction(
    n_periods=12,
    remainder="drop",
    column="time_index",
    input_range=(1, 365),
)
rbf.fit(df)
rbf_features = rbf.transform(df)
print(rbf_features.shape)
fig, ax = plt.subplots(nrows=1, ncols=1)
pd.DataFrame(rbf_features).plot(ax=ax)
fig.savefig('rbf_features.png')

# Use rbf features in a pipeline.
pipeline_rbf = Pipeline(
    [
        ("basis_funcs", RepeatingBasisFunction(
            n_periods=12,
            remainder="drop",
            column="time_index",
            input_range=(1, 365),
        )),
        ("regression", LinearRegression()),
    ]
)
pred_rbf = pipeline_rbf.fit(df, df.rentals).predict(df)

# Plot predictions and actual time series
fig, ax = plt.subplots(nrows=1, ncols=1)
df.plot(kind="scatter", x="time_index", y="rentals", ax=ax)
plt.plot(df.time_index, pred_rbf, color="orange")
fig.savefig("rental_rbf.png")