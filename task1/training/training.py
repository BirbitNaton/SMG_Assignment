import hashlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from task1.data.preprocessing import revert_target
from task1.data.utils import DATA_PATH
from task1.training.models import TrainingParamsUtils


def log_data_version(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    log_entire_dataset: bool = False,
):
    """Log data versioning information.

    Logs various metadata about the training and testing datasets,
    including their versions, sizes, and feature counts.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training target.
    X_test : pd.DataFrame
        The testing features.
    y_test : pd.Series
        The testing target.
    random_state : int
        The random state for reproducibility.
    """
    # Log data versioning
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(
            pd.concat((X_train, y_train), axis=1), index=True
        ).values
    ).hexdigest()
    mlflow.log_param("data_version", data_hash)
    mlflow.log_param("n_train_samples", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("random_state", random_state)

    # Log datasets as artifacts (CSV)
    if log_entire_dataset:
        X_train_path = DATA_PATH / "X_train.csv"
        y_train_path = DATA_PATH / "y_train.csv"
        X_test_path = DATA_PATH / "X_test.csv"
        y_test_path = DATA_PATH / "y_test.csv"
        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        mlflow.log_artifact(X_train_path, artifact_path="dataset")
        mlflow.log_artifact(y_train_path, artifact_path="dataset")
        mlflow.log_artifact(X_test_path, artifact_path="dataset")
        mlflow.log_artifact(y_test_path, artifact_path="dataset")


def log_model_metrics(y_test: pd.Series, y_pred: np.ndarray, scores: np.ndarray):
    """Log model evaluation metrics.

    Logs evaluation metrics for the model, including R2, MAE, MSE, RMSE, and MAPE.

    Parameters
    ----------
    y_test : pd.Series
        original target values for the test set
    y_pred : np.ndarray
        predicted target values from the model for the test set
    scores : np.ndarray
        cross-validation scores.
    """
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mlflow.log_metric("cv_mean_r2", scores.mean())
    mlflow.log_metric("cv_std_r2", scores.std())
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mape", mape)


def start_training(
    pipeline: Pipeline,
    training_parameters: TrainingParamsUtils,
):
    """Start training process with MLflow tracking.

    This function trains the given pipeline on the provided training data,
    evaluates it using cross-validation, tests on the test set, and logs
    all relevant information to MLflow.

    Parameters
    ----------
    pipeline : Pipeline
        a sklearn Pipeline object containing the model.
        The pipeline could also contain preprocessing steps,
        but the functions used have to be compatible with the pipeline.

    training_parameters : TrainingParamsUtils
        An utility class containing all training parameters, given as such:
            X_train : pd.DataFrame
                training features.
            y_train : pd.Series
                training target.
            X_test : pd.DataFrame
                testing features.
            y_test : pd.Series
                testing target.
            random_state : int
                random state for reproducibility, by default 42
            cv_splits : int
                number of cross-validation splits, by default 5
            scoring : str
                scoring metric for cross-validation, by default "r2"
            description : str | None, optional
                description of the run, by default None
            tags : dict | None, optional
                tags to apply to the run, by default None
    """
    mlflow.set_experiment(training_parameters.experiment_name)

    with mlflow.start_run(
        run_name="catboost_pipeline",
        log_system_metrics=True,
        description=training_parameters.description,
        tags=training_parameters.tags,
    ):
        # Log data versioning
        log_data_version(
            training_parameters.X_train,
            training_parameters.y_train,
            training_parameters.X_test,
            training_parameters.y_test,
            training_parameters.random_state,
            training_parameters.log_entire_dataset,
        )

        # Evaluate with CV
        cv = KFold(
            n_splits=training_parameters.cv_splits,
            shuffle=True,
            random_state=training_parameters.random_state,
        )
        scores = cross_val_score(
            pipeline,
            training_parameters.X_train,
            training_parameters.y_train,
            cv=cv,
            scoring=training_parameters.scoring,
        )

        # Fit the pipeline
        pipeline.fit(training_parameters.X_train, training_parameters.y_train)

        # Log metrics - CV and test
        y_pred = revert_target(pipeline.predict(training_parameters.X_test))
        y_test = revert_target(training_parameters.y_test)
        log_model_metrics(y_test, y_pred, scores)

        # Log hyperparameters (example: CatBoost settings)
        mlflow.log_params(pipeline.named_steps["model"].get_params())

        # Log final model artifact
        mlflow.sklearn.log_model(pipeline, "model")


def register_model(model_name: str, model_uri: str = "runs:/{run_id}/model"):
    """
    Register the trained model in MLflow Model Registry.

    Args:
        model_name : str
            the name to register the model under.
        model_uri : str
            the URI of the model to register.
    """
    mlflow.register_model(model_uri=model_uri, name=model_name)
