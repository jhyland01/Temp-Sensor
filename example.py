# +
# # !python -m pip install -U scikit-learn
# -

import zipfile
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path_to_data_zip: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    with zipfile.ZipFile(path_to_data_zip) as zf:
        with zf.open("X_train.csv") as f:
            X_train = pd.read_csv(f, index_col=0, parse_dates=True)
        with zf.open("y_train.csv") as f:
            y_train = pd.read_csv(f, index_col=0, parse_dates=True).squeeze()
        with zf.open("X_test.csv") as f:
            X_test = pd.read_csv(f, index_col=0, parse_dates=True)
        with zf.open("y_test.csv") as f:
            y_test = pd.read_csv(f, index_col=0, parse_dates=True).squeeze()
    return X_train, y_train, X_test, y_test


def evaluate(y_test: pd.Series, y_predict: pd.Series, show_corr: bool = True) -> None:
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    r2 = r2_score(y_test, y_predict)
    print(f"The root mean squared error (RMSE) on test set  : {rmse:.4f}")
    print(f"The coefficient of determination (R2) on test set: {r2:.4f}")
    if show_corr:
        (
            pd.DataFrame({"actual": y_test, "predicted": y_predict})
            .assign(absolute_difference=lambda d: (d["actual"] - d["predicted"]).abs())
            .plot.scatter(x="actual", y="predicted", c="absolute_difference", figsize=(8, 8), grid=True)
        )
    return rmse, r2


X_train, y_train, X_test, y_test = load_data(r"C:\Users\gcalvo\Downloads\single_turbine_data.zip")
model = make_pipeline(SimpleImputer(), LinearRegression())
model.fit(X_train, y_train)
mse, r2 = evaluate(y_test=y_test, y_predict=model.predict(X_test))


