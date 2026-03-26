import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate classification model performance."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class label names
        average : str
            Averaging method for metrics

        Returns:
        --------
        dict : Evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns:
        --------
        np.ndarray
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray
    ) -> str:
        """
        Get detailed classification report.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class names

        Returns:
        --------
        str
            Classification report
        """
        return classification_report(y_true, y_pred, target_names=class_names)

    @staticmethod
    def get_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute per-class metrics.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class names

        Returns:
        --------
        pd.DataFrame
            Per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics_df = pd.DataFrame({
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        return metrics_df

    @staticmethod
    def print_evaluation_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray,
        dataset_name: str = "Test Set"
    ) -> None:
        """
        Print comprehensive evaluation report.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class names
        dataset_name : str
            Name of dataset being evaluated
        """
        metrics = ModelEvaluator.evaluate(y_true, y_pred, class_names)
        conf_matrix = ModelEvaluator.get_confusion_matrix(y_true, y_pred)
        class_report = ModelEvaluator.get_classification_report(
            y_true, y_pred, class_names
        )
        per_class = ModelEvaluator.get_per_class_metrics(y_true, y_pred, class_names)

        print("\n" + "="*80)
        print(f"MODEL EVALUATION - {dataset_name.upper()}")
        print("="*80)

        print("\n" + "-"*80)
        print("OVERALL METRICS")
        print("-"*80)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")

        print("\n" + "-"*80)
        print("PER-CLASS METRICS")
        print("-"*80)
        print(per_class.to_string(index=False))

        print("\n" + "-"*80)
        print("CONFUSION MATRIX")
        print("-"*80)
        print(f"Shape: {conf_matrix.shape}")
        print("\nMatrix:")
        print(conf_matrix)

        print("\n" + "-"*80)
        print("CLASSIFICATION REPORT")
        print("-"*80)
        print(class_report)

        print("="*80 + "\n")

    @staticmethod
    def get_prediction_accuracy_per_class(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray
    ) -> pd.DataFrame:
        """
        Get per-class accuracy.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class names

        Returns:
        --------
        pd.DataFrame
            Per-class accuracy
        """
        conf_matrix = confusion_matrix(y_true, y_pred)

        per_class_accuracy = []
        for i in range(len(class_names)):
            if conf_matrix[i].sum() > 0:
                accuracy = conf_matrix[i][i] / conf_matrix[i].sum()
            else:
                accuracy = 0.0

            per_class_accuracy.append({
                "class": class_names[i],
                "accuracy": float(accuracy),
                "correct": int(conf_matrix[i][i]),
                "total": int(conf_matrix[i].sum())
            })

        return pd.DataFrame(per_class_accuracy)

    @staticmethod
    def print_confusion_matrix_detailed(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: np.ndarray
    ) -> None:
        """
        Print detailed confusion matrix.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : np.ndarray
            Class names
        """
        conf_matrix = confusion_matrix(y_true, y_pred)

        print("\n" + "="*80)
        print("DETAILED CONFUSION MATRIX")
        print("="*80)

        # Header
        print("\nRows: True Label | Columns: Predicted Label\n")

        # Column headers
        header = "            " + "".join([f"{name:>8}" for name in class_names])
        print(header)

        # Matrix rows
        for i, true_class in enumerate(class_names):
            row_str = f"{true_class:<12}"
            for j in range(len(class_names)):
                row_str += f"{conf_matrix[i][j]:>8}"
            print(row_str)

        print("\n" + "="*80 + "\n")
