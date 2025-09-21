import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from task1.training.models import TrainingParamsUtils
from task1.training.training import log_data_version, log_model_metrics, start_training


def test_data_logging():
    """Test data logging functionality.

    Runs mlflow logging to ensure its availability.
    """
    log_data_version(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 42)


def test_metrics_logging():
    """Test metrics logging functionality.

    Runs mlflow logging to ensure metrics can be calculated and logged.
    """
    log_model_metrics(pd.Series([0]), np.array([0]), np.array([0]))


def test_training():
    """Tests training pipeline.

    Mocks most of the fields and validates pipelines' robustness an OLS model.
    """
    mlflow.end_run()
    start_training(
        Pipeline(steps=[("model", LinearRegression())]),
        TrainingParamsUtils(
            X_train=pd.DataFrame({"feat_1": list(range(40))}),
            y_train=pd.Series(list(range(40))),
            X_test=pd.DataFrame({"feat_1": list(range(40))}),
            y_test=pd.Series(list(range(40))),
            random_state=42,
            cv_splits=2,
            scoring="r2",
            experiment_name="test_exp_name",
            log_entire_dataset=False,
        ),
    )
    mlflow.end_run()
