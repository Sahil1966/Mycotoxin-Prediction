import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessing")

class DataPreprocessor:
    """Class for loading and preprocessing the dataset."""

    def __init__(self, data_path):
        """
        Initialize the DataPreprocessor class.

        Parameters:
        - data_path (str): Path to the dataset CSV file.
        """
        self.data_path = data_path
        self.df = None  # Will hold the loaded dataset

    def load_data(self):
        """Loads the dataset from the given path."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def explore_data(self):
        """Displays dataset information and summary statistics."""
        if self.df is None:
            logger.error("Data not loaded. Run `load_data()` first.")
            return

        logger.info("Displaying dataset info:")
        print(self.df.info())

        logger.info("Summary statistics:")
        print(self.df.describe())

        logger.info("Checking for missing values:")
        print(f"Missing values are : \n{self.df.isnull().sum()}")

    def preprocess_data(self):
        """Performs preprocessing: handling missing values, feature selection."""
        if self.df is None:
            logger.error("No data loaded. Load data first.")
            return

        # Drop irrelevant columns
        if 'hsi_id' in self.df.columns:
            self.df.drop(columns=['hsi_id'], inplace=True)
            logger.info("Dropped irrelevant column: hsi_id")

        # Select only numeric columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns

        # Fill missing values with median
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        logger.info("Missing values handled.")

        return self.df
