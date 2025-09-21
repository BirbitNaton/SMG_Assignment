import pandas as pd
from pydantic import BaseModel, ConfigDict


class TrainingParamsUtils(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    random_state: int
    cv_splits: int
    scoring: str
    experiment_name: str
    log_entire_dataset: bool = False
    description: str | None = None
    tags: dict | None = None
