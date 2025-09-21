import argparse
import copy
import json
from pathlib import Path

import mlflow
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from task1.data.dataloader import CustomDataLoader
from task1.data.preprocessing import convert_target, get_preprocessing_stages
from task1.training.models import TrainingParamsUtils
from task1.training.training import register_model, start_training

parser = argparse.ArgumentParser(description="Train and register CatBoost model")
parser.add_argument(
    "--settings_json_path",
    type=str,
    default="default_settings.json",
    help="Path to the settings JSON file name (default: default_settings.json)",
)
RANDNOM_STATE = 42
CV_SPLITS = 5
DEFAULT_SCORING = "r2"
DEFAULT_EXPERIMENT_NAME = "Default Experiment"

if __name__ == "__main__":
    # load data
    data_loader = CustomDataLoader()
    df = data_loader.load_data()
    X = copy.deepcopy(df)
    y = copy.deepcopy(convert_target(data_loader.target_column()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDNOM_STATE
    )

    # create preprocessing pipeline
    preprocessing_stages = get_preprocessing_stages()

    # preprocess data
    X_train = Pipeline(steps=preprocessing_stages).fit_transform(X_train)
    X_test = Pipeline(steps=preprocessing_stages).fit_transform(X_test)

    # load training parameters
    args = parser.parse_args()
    settings_path = (
        Path(__file__).parent.parent / "model_settings" / args.settings_json_path
    )
    parameters = json.load(settings_path.open())
    parameters["random_seed"] = parameters.get("random_seed", RANDNOM_STATE)
    training_parameters = TrainingParamsUtils(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=parameters["random_seed"],
        cv_splits=parameters.pop("cv_splits", CV_SPLITS),
        scoring=parameters.pop("scoring", DEFAULT_SCORING),
        experiment_name=parameters.pop("experiment_name", DEFAULT_EXPERIMENT_NAME),
        description=parameters.pop("description", None),
        tags=parameters.pop("tags", None),
        log_entire_dataset=parameters.pop("log_entire_dataset", False),
    )

    # create a pipeline with model
    pipeline = Pipeline(
        steps=[
            (
                "model",
                CatBoostRegressor(**parameters),
            ),
        ]
    )

    # start training
    start_training(pipeline, training_parameters)

    # register the model
    best_run = mlflow.search_runs(order_by=["metrics.cv_mean_r2 DESC"]).iloc[0]
    register_model(
        model_name="CatBoost_House_Price_Predictor",
        model_uri=f"runs:/{best_run.run_id}/model",
    )
