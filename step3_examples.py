"""
Step 3 Examples: Using Genre Classification Model

Demonstrates various use cases for training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from model_training import GenreClassificationModel
from model_evaluation import ModelEvaluator
from model_pipeline import ClassificationPipeline
from config import GENRES_DIR


# ==============================================================================
# EXAMPLE 1: Basic Train-Test Split and Evaluation
# ==============================================================================
def example_1_basic_training():
    """Train model with 80-20 split and print metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training and Evaluation")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    # Initialize model
    model = GenreClassificationModel(random_state=42)

    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(df_features, test_size=0.2)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model
    model.train(X_train, y_train, n_estimators=100)
    print("✓ Model trained")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    return model, X_test, y_test


# ==============================================================================
# EXAMPLE 2: Cross-Validation Analysis
# ==============================================================================
def example_2_cross_validation():
    """5-fold cross-validation to assess model robustness."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Cross-Validation")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    pipeline = ClassificationPipeline(random_state=42)
    cv_results = pipeline.cross_validate(df_features, n_splits=5, n_estimators=100)

    print(f"CV Accuracy scores: {cv_results['cv_accuracies']}")
    print(f"Mean CV Accuracy: {np.mean(cv_results['cv_accuracies']):.4f} "
          f"(+/- {np.std(cv_results['cv_accuracies']):.4f})")
    print(f"CV F1 scores: {cv_results['cv_f1_scores']}")
    print(f"Mean CV F1: {np.mean(cv_results['cv_f1_scores']):.4f}")

    return cv_results


# ==============================================================================
# EXAMPLE 3: Feature Importance Analysis
# ==============================================================================
def example_3_feature_importance():
    """Identify most important features for classification."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Feature Importance Analysis")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    model = GenreClassificationModel(random_state=42)
    X_train, X_test, y_train, y_test = model.prepare_data(df_features, test_size=0.2)
    model.train(X_train, y_train, n_estimators=100)

    # Get top 15 important features
    importance_df = model.get_feature_importance(top_n=15)
    print("\nTop 15 Most Important Features:")
    print(importance_df.to_string(index=False))

    return importance_df


# ==============================================================================
# EXAMPLE 4: Model Persistence
# ==============================================================================
def example_4_model_persistence():
    """Train, save, and reload model."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Persistence (Save/Load)")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    # Train and save
    model = GenreClassificationModel(random_state=42)
    X_train, X_test, y_train, y_test = model.prepare_data(df_features, test_size=0.2)
    model.train(X_train, y_train, n_estimators=100)

    model_path = GENRES_DIR.parent / "example_model.pkl"
    model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")

    # Load model
    loaded_model = GenreClassificationModel()
    loaded_model.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")

    # Test loaded model
    y_pred_loaded = loaded_model.predict(X_test)
    accuracy_loaded = np.mean(y_pred_loaded == y_test)
    print(f"Loaded Model Accuracy: {accuracy_loaded:.4f}")

    # Clean up
    model_path.unlink()
    print("✓ Example model cleaned up")

    return loaded_model


# ==============================================================================
# EXAMPLE 5: Misclassified Samples Analysis
# ==============================================================================
def example_5_misclassified_samples():
    """Find high-confidence misclassifications."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Misclassified Samples Analysis")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    pipeline = ClassificationPipeline(random_state=42)
    pipeline.run_full_pipeline(df_features, test_size=0.2, n_estimators=100, save_model=False)

    misclassified = pipeline.get_misclassified_samples(df_features, top_n=5)
    print("\nTop 5 High-Confidence Misclassifications:")
    print(misclassified)

    return misclassified


# ==============================================================================
# EXAMPLE 6: Probability Predictions
# ==============================================================================
def example_6_probability_predictions():
    """Get class probability predictions."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Probability Predictions")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    model = GenreClassificationModel(random_state=42)
    X_train, X_test, y_train, y_test = model.prepare_data(df_features, test_size=0.2)
    model.train(X_train, y_train, n_estimators=100)

    # Get probabilities for first 5 test samples
    proba = model.predict_proba(X_test[:5])
    print("\nClass probabilities for first 5 test samples:")
    print(proba)

    return proba


# ==============================================================================
# EXAMPLE 7: Hyperparameter Effects
# ==============================================================================
def example_7_hyperparameter_effects():
    """Compare different RandomForest hyperparameters."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Hyperparameter Effects")
    print("="*80)

    features_path = GENRES_DIR.parent / "features_extracted.csv"
    df_features = pd.read_csv(features_path)

    model = GenreClassificationModel(random_state=42)
    X_train, X_test, y_train, y_test = model.prepare_data(df_features, test_size=0.2)

    print("\nComparing different n_estimators values:")
    for n_est in [50, 100, 200, 300]:
        model_temp = GenreClassificationModel(random_state=42)
        model_temp.train(X_train, y_train, n_estimators=n_est)
        y_pred = model_temp.predict(X_test)
        acc = np.mean(y_pred == y_test)
        print(f"  n_estimators={n_est:3d}: Accuracy = {acc:.4f}")

    return None


# ==============================================================================
# MAIN: Run all examples
# ==============================================================================
def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("STEP 3: MACHINE LEARNING EXAMPLES")
    print("="*80)

    try:
        example_1_basic_training()
        example_2_cross_validation()
        example_3_feature_importance()
        example_4_model_persistence()
        example_5_misclassified_samples()
        example_6_probability_predictions()
        example_7_hyperparameter_effects()

        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please ensure Step 2 (feature extraction) is completed first.")


if __name__ == "__main__":
    main()
