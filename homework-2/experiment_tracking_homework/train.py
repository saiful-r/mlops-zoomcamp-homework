import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_experiment('homework-2')


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():

        mlflow.set_tag("model", "RandomForestRegressor")
        mlflow.set_tag("dataset", "NYC taxi")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("project", "homework 2")
        mlflow.sklearn.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        mlflow.log_param("train-data-path", os.path.join(data_path, "train.pkl"))
        mlflow.log_param("val-data-path", os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        # mlflow.log_param("max_depth", 10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
