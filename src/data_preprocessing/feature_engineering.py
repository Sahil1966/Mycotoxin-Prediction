import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib  # Import joblib to save the scaler
import os  # For handling file paths
from src.utils.logger import get_logger  # Import the logger

logger = get_logger("FeatureEngineering")  # Create a logger for this class

class FeatureEngineering:
    def __init__(self, df):
        """Initialize with a DataFrame."""
        self.df = df.copy()
        self.scaler = None  # Placeholder for the scaler
    
    # Step 1: Boxplots Before Outlier Removal
    def plot_boxplots(self, title="Boxplots", batch_size=25):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        num_batches = (len(numeric_cols) + batch_size - 1) // batch_size  # Calculate number of batches

        # Create a single figure for all boxplots
        fig, axes = plt.subplots(nrows=num_batches, ncols=1, figsize=(12, num_batches * 4), squeeze=False)
        
        for i in range(num_batches):
            batch_cols = numeric_cols[i * batch_size:(i + 1) * batch_size]
            
            # Create a boxplot for the current batch of columns
            self.df[batch_cols].plot(kind="box", ax=axes[i, 0], subplots=False)
            axes[i, 0].set_title(f"{title} - Batch {i + 1}")
            axes[i, 0].set_ylabel('Values')
        
        plt.tight_layout()
        plt.show()
        
    # Step 2: Outlier Removal Using IQR
    def remove_outliers_iqr(self):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        logger.info("Outliers removed using IQR method.")
        return self.df

    # Step 3: Correlation Heatmap for Feature Selection
    def plot_correlation_heatmap(self):
        corr_matrix = self.df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        plt.show()
        logger.info("Correlation heatmap plotted.")

    # Step 4: Feature Selection Using Elastic Net (with scaler saving)
    def elastic_net_feature_selection(self, target_col='vomitoxin_ppb', l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0], 
                                       cv=5, max_iter=1000, save_scaler=False):
        """
        Perform feature selection using Elastic Net without manual correlation thresholds.
        
        Args:
            target_col: Name of the target column.
            l1_ratios: List of L1/L2 mixing ratios (0= Ridge, 1= Lasso).
            cv: Number of cross-validation folds.
            max_iter: Maximum number of iterations for convergence.
            save_scaler: Boolean flag to save the scaler as a .pkl file.
        
        Returns:
            DataFrame with selected features + target.
        """
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Create a pipeline for scaling and Elastic Net
        pipeline = Pipeline([ 
            ('scaler', StandardScaler()),  # Standardize features
            ('enet', ElasticNetCV(
                l1_ratio=l1_ratios,  # Test different L1/L2 balances
                cv=cv,               # Cross-validation folds
                n_jobs=-1,          # Use all cores
                random_state=42,
                max_iter=max_iter    # Set maximum iterations
            ))
        ])
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(X, y)
        
        # Extract the scaler from the pipeline
        self.scaler = pipeline.named_steps['scaler']
        
        # Optionally save the scaler as a .pkl file
        if save_scaler:
            self.save_scaler()

        # Extract selected features (non-zero coefficients)
        selected_features = X.columns[pipeline.named_steps['enet'].coef_ != 0].tolist()
        
        # Log selected features for feedback
        logger.info(f"Selected features: {selected_features}")
        
        # Return DataFrame with selected features + target
        return self.df[selected_features + [target_col]]
    
    # Method to save the scaler as a .pkl file
    def save_scaler(self, filename="scaler.pkl"):
        """
        Save the scaler to a .pkl file in the models/src/ directory.
        """
        # Create the directory if it doesn't exist
        os.makedirs('models/src', exist_ok=True)
        
        # Define the path to save the scaler
        scaler_path = os.path.join('models', 'src', filename)
        
        # Check if the scaler is available
        if self.scaler:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        else:
            logger.warning("Scaler is not fitted yet, cannot save.")
    
    def run_feature_engineering(self):
        logger.info("Starting feature engineering process.")

        # Plot boxplots before outlier removal
        self.plot_boxplots(title="Boxplots Before Outlier Removal")

        # Remove outliers
        self.df = self.remove_outliers_iqr()
        logger.info(f"Number of features after outlier removal: {self.df.shape[1]}")

        # Plot boxplots after outlier removal
        self.plot_boxplots(title="Boxplots After Outlier Removal")

        # Plot correlation heatmap and identify correlated features
        self.plot_correlation_heatmap()

        # Perform Elastic Net feature selection and save the scaler
        self.df = self.elastic_net_feature_selection(target_col='vomitoxin_ppb', save_scaler=True)  # Set save_scaler=True to save the scaler
        logger.info(f"Number of features after Elastic Net feature selection: {self.df.shape[1]}")
        logger.info("Feature engineering process completed.")