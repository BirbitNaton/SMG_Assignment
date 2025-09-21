import json
from pathlib import Path

import pandas as pd

from task1.data.utils import DEFAULT_DATA_PATH, DEFAULT_SCHEMA_PATH


class CustomDataLoader:
    """Custom data loader for loading and processing dataset.

    If no paths are provided, it defaults to loading from the `data` directory.

    Attributes:
        data_path (Path): Path to the CSV data file
        schema_path (Path): Path to the JSON schema file

    Methods:
        load_data(): Loads the dataset into a pandas DataFrame
        target_column(): Returns the target column as specified in the schema
    """

    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
    ):
        self.data_path = data_path
        self.schema_path = schema_path
        self.df = pd.read_csv(self.data_path)
        self.dataset_scheme = json.load(self.schema_path.open())

    def load_data(self):
        """Load the dataset into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        return self.df.drop(columns=[self.dataset_scheme["target"]])

    def target_column(self):
        """Get the target column from the dataset.

        Reads the target column name from the dataset schema and returns
        corresponding column from the DataFrame.

        Returns
        -------
        pd.Series
            The target column.
        """
        return self.df[self.dataset_scheme["target"]]
