import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenreClassificationModel:
    """Train and manage genre classification model."""

    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.class_labels = None
        self.training_info = {}

    def prepare_data(
        self,
        df_features: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """
        Prepare data for training.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame from feature extraction
        test_size : float
            Proportion of data for testing

        Returns:
        --------
        X_train, X_test, y_train, y_test, label_encoder
        """
        # Extract features and labels
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        
        X = df_features[feature_cols].values
        y = df_features["genre"].values

        self.feature_columns = feature_cols

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        self.label_encoder = label_encoder
        self.class_labels = label_encoder.classes_

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        logger.info(f"Data prepared:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Classes: {len(self.class_labels)}")

        return X_train, X_test, y_train, y_test, label_encoder

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.

        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        min_samples_leaf : int
            Minimum samples per leaf
        n_jobs : int
            Number of parallel jobs (-1 for all)

        Returns:
        --------
        RandomForestClassifier
            Trained model
        """
        logger.info("Training Random Forest Classifier...")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=1
        )

        self.model.fit(X_train, y_train)

        self.training_info = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        }

        logger.info("Model training complete")

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters:
        -----------
        X : np.ndarray
            Features

        Returns:
        --------
        np.ndarray
            Predicted labels (encoded)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters:
        -----------
        X : np.ndarray
            Features

        Returns:
        --------
        np.ndarray
            Probability estimates
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return feature_importance_df.head(top_n)

    def save_model(self, filepath: Path) -> None:
        """
        Save trained model to disk.

        Parameters:
        -----------
        filepath : Path
            Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "class_labels": self.class_labels,
            "training_info": self.training_info
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        """
        Load model from disk.

        Parameters:
        -----------
        filepath : Path
            Path to model file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_columns = model_data["feature_columns"]
        self.class_labels = model_data["class_labels"]
        self.training_info = model_data["training_info"]

        logger.info(f"Model loaded from {filepath}")
