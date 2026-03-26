import pandas as pd
import numpy as np
from pathlib import Path
import logging

from model_training import GenreClassificationModel
from model_evaluation import ModelEvaluator
from config import GENRES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """Complete model training and evaluation pipeline."""

    def __init__(self, random_state: int = 42):
        """
        Initialize pipeline.

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = GenreClassificationModel(random_state=random_state)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None

    def run_full_pipeline(
        self,
        df_features: pd.DataFrame,
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = None,
        save_model: bool = True,
        model_path: Path = None
    ) -> dict:
        """
        Run complete classification pipeline.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame from Step 2
        test_size : float
            Test set proportion
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum tree depth
        save_model : bool
            Whether to save model
        model_path : Path, optional
            Path to save model

        Returns:
        --------
        dict : Pipeline results
        """
        logger.info("Starting classification pipeline...")

        # Step 1: Prepare data
        print("\n[1/4] Preparing data...")
        self.X_train, self.X_test, self.y_train, self.y_test, _ = \
            self.model.prepare_data(df_features, test_size=test_size)

        # Step 2: Train model
        print("\n[2/4] Training model...")
        self.model.train(
            self.X_train,
            self.y_train,
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        # Step 3: Evaluate model
        print("\n[3/4] Evaluating model...")
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

        self.print_evaluation_report()

        # Step 4: Feature importance
        print("\n[4/4] Computing feature importance...")
        feature_importance = self.model.get_feature_importance(top_n=20)
        print("\nTop 20 Important Features:")
        print("-"*60)
        print(feature_importance.to_string(index=False))

        # Save model
        if save_model:
            if model_path is None:
                model_path = GENRES_DIR.parent / "trained_model.pkl"

            self.model.save_model(model_path)
            print(f"\n✓ Model saved to {model_path}")

        logger.info("Pipeline complete!")

        return {
            "model": self.model,
            "train_accuracy": float(
                np.mean(self.y_pred_train == self.y_train)
            ),
            "test_accuracy": float(
                np.mean(self.y_pred_test == self.y_test)
            ),
            "feature_importance": feature_importance
        }

    def print_evaluation_report(self) -> None:
        """Print comprehensive evaluation report."""
        print("\n" + "="*80)
        print("MODEL EVALUATION REPORT")
        print("="*80)

        # Training set evaluation
        ModelEvaluator.print_evaluation_report(
            self.y_train,
            self.y_pred_train,
            self.model.class_labels,
            dataset_name="Training Set"
        )

        # Test set evaluation
        ModelEvaluator.print_evaluation_report(
            self.y_test,
            self.y_pred_test,
            self.model.class_labels,
            dataset_name="Test Set"
        )

        # Detailed confusion matrix
        ModelEvaluator.print_confusion_matrix_detailed(
            self.y_test,
            self.y_pred_test,
            self.model.class_labels
        )

        # Per-class accuracy
        per_class_acc = ModelEvaluator.get_prediction_accuracy_per_class(
            self.y_test,
            self.y_pred_test,
            self.model.class_labels
        )

        print("-"*80)
        print("PER-CLASS ACCURACY (Test Set)")
        print("-"*80)
        print(per_class_acc.to_string(index=False))
        print()

    def get_misclassified_samples(
        self,
        df_features: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get most confidently misclassified samples.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Original feature DataFrame
        top_n : int
            Number of samples to return

        Returns:
        --------
        pd.DataFrame
            Misclassified samples
        """
        # Get test set indices
        test_indices = range(len(self.X_train), len(self.X_train) + len(self.X_test))

        # Get probabilities
        y_proba = self.model.predict_proba(self.X_test)

        # Find misclassified
        misclassified_mask = self.y_pred_test != self.y_test
        misclassified_proba = y_proba[misclassified_mask]
        misclassified_pred = self.y_pred_test[misclassified_mask]
        misclassified_true = self.y_test[misclassified_mask]

        # Get confidence (max probability)
        confidence = np.max(misclassified_proba, axis=1)

        # Sort by confidence
        sorted_indices = np.argsort(confidence)[::-1][:top_n]

        misclassified_data = []
        for idx in sorted_indices:
            test_idx = np.where(misclassified_mask)[0][idx]

            misclassified_data.append({
                "true_genre": self.model.class_labels[misclassified_true[idx]],
                "predicted_genre": self.model.class_labels[misclassified_pred[idx]],
                "confidence": float(confidence[idx])
            })

        return pd.DataFrame(misclassified_data)

    def cross_validate(
        self,
        df_features: pd.DataFrame,
        n_splits: int = 5,
        n_estimators: int = 100
    ) -> dict:
        """
        Perform k-fold cross-validation.

        Parameters:
        -----------
        df_features : pd.DataFrame
            Feature DataFrame
        n_splits : int
            Number of folds
        n_estimators : int
            Number of trees for RF

        Returns:
        --------
        dict : Cross-validation results
        """
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

        # Prepare data
        feature_cols = [col for col in df_features.columns 
                       if col not in ["file_path", "genre"]]
        X = df_features[feature_cols].values
        y = pd.factorize(df_features["genre"])[0]

        # Cross-validation
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=self.random_state)

        # Create fresh RF model for cross-validation
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted'
        }

        cv_results = cross_validate(
            rf_model,
            X, y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1
        )

        cv_accuracies = cv_results['test_accuracy']
        cv_f1_scores = cv_results['test_f1_weighted']

        logger.info(f"Cross-validation (k={n_splits}):")
        logger.info(f"  Mean Accuracy: {np.mean(cv_accuracies):.4f} "
                    f"(+/- {np.std(cv_accuracies):.4f})")
        logger.info(f"  Mean F1: {np.mean(cv_f1_scores):.4f}")

        return {
            "cv_accuracies": cv_accuracies.tolist(),
            "cv_f1_scores": cv_f1_scores.tolist(),
            "accuracy_mean": float(np.mean(cv_accuracies)),
            "accuracy_std": float(np.std(cv_accuracies)),
            "f1_mean": float(np.mean(cv_f1_scores)),
            "f1_std": float(np.std(cv_f1_scores))
        }
