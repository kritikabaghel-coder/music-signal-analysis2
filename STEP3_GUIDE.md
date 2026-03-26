"""
STEP 3: MACHINE LEARNING - GENRE CLASSIFICATION MODEL
========================================================

This module implements Random Forest classification for GTZAN music genres.

## OVERVIEW
-----------

Step 3 trains a machine learning model to classify music by genre using features
extracted in Step 2. The pipeline includes:

1. Data preparation with stratified train-test split
2. Random Forest classifier training
3. Model evaluation with multiple metrics
4. Feature importance analysis
5. Model persistence and loading

## CONSTRAINTS
--------------

✓ No deep learning - using scikit-learn only
✓ Modular, clean code with object-oriented design
✓ Full reproducibility with random_state=42
✓ Stratified split to maintain class distribution


## KEY COMPONENTS
-----------------

### 1. GenreClassificationModel (model_training.py)
   
   Purpose: Train and manage the Random Forest model
   
   Key Methods:
   - prepare_data(df_features, test_size=0.2)
     → Splits data into train/test with stratified sampling
     → Encodes genre labels using LabelEncoder
     → Returns: X_train, X_test, y_train, y_test, label_encoder
   
   - train(X_train, y_train, n_estimators=100, max_depth=None)
     → Trains RandomForestClassifier with given parameters
     → Stores training metadata
     → Returns: trained RandomForestClassifier
   
   - predict(X)
     → Makes class predictions on new data
     → Returns: array of predicted class indices
   
   - predict_proba(X)
     → Returns probability estimates for each class
     → Useful for confidence analysis
   
   - get_feature_importance(top_n=20)
     → Ranks features by importance
     → Returns: DataFrame with feature names and importance scores
   
   - save_model(filepath) / load_model(filepath)
     → Persists model using joblib
     → Saves: model, label encoder, feature columns, metadata


### 2. ModelEvaluator (model_evaluation.py)
   
   Purpose: Compute comprehensive evaluation metrics
   
   Static Methods:
   - evaluate(y_true, y_pred, class_names)
     → Basic metrics: accuracy, precision, recall, f1
   
   - get_confusion_matrix(y_true, y_pred)
     → Returns confusion matrix array
   
   - get_classification_report(y_true, y_pred, class_names)
     → Standard sklearn classification report
   
   - get_per_class_metrics(y_true, y_pred, class_names)
     → DataFrame with per-class precision, recall, f1
   
   - print_evaluation_report(y_true, y_pred, class_names, dataset_name)
     → Formatted console output of all metrics
   
   - print_confusion_matrix_detailed(y_true, y_pred, class_names)
     → Confusion matrix with genre labels


### 3. ClassificationPipeline (model_pipeline.py)
   
   Purpose: Orchestrate full training workflow
   
   Key Methods:
   - run_full_pipeline(df_features, test_size=0.2, n_estimators=100, ...)
     → Complete workflow: prepare → train → evaluate → save
     → Returns: dictionary with accuracy, feature importance, etc.
   
   - print_evaluation_report()
     → Prints metrics for train and test sets
   
   - get_misclassified_samples(df_features, top_n=10)
     → Finds high-confidence misclassifications
     → Useful for error analysis
   
   - cross_validate(df_features, n_splits=5, n_estimators=100)
     → K-fold cross-validation for robustness assessment
     → Returns: fold scores, means, stds


## EXECUTION WORKFLOW
---------------------

### Basic Training (train_model.py)

    from model_pipeline import ClassificationPipeline
    import pandas as pd
    
    # 1. Load features from Step 2
    df_features = pd.read_csv("features_extracted.csv")
    
    # 2. Create pipeline
    pipeline = ClassificationPipeline(random_state=42)
    
    # 3. Train model (saves automatically)
    results = pipeline.run_full_pipeline(
        df_features,
        test_size=0.2,
        n_estimators=100,
        save_model=True,
        model_path="trained_model.pkl"
    )
    
    # 4. Results contain:
    #    - results['train_accuracy']: Training accuracy
    #    - results['test_accuracy']: Test accuracy
    #    - results['feature_importance']: Top features


### Inference with Saved Model

    from model_training import GenreClassificationModel
    
    # 1. Load model
    model = GenreClassificationModel()
    model.load_model("trained_model.pkl")
    
    # 2. Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # 3. Get feature importance
    importance = model.get_feature_importance(top_n=15)


## EVALUATION METRICS
-------------------

### Overall Metrics
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP) - how many predicted positives are correct
- Recall: TP / (TP + FN) - how many actual positives are found
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

### Per-Class Metrics
For each genre, we compute separate precision, recall, f1 scores
to identify which genres are harder to classify.

### Confusion Matrix
Shows true vs predicted classifications for each class
- Diagonal values = correct predictions
- Off-diagonal = misclassifications
- Helps identify which genres are confused with each other


## HYPERPARAMETERS
------------------

### RandomForestClassifier
- n_estimators: 100 (default)
  Number of trees. Higher = better but slower.
  Typical range: 50-500

- max_depth: None (default)
  Maximum tree depth. None = unlimited.
  Lower = less overfitting, but may underfit.

- min_samples_split: 2 (default)
  Minimum samples required to split a node.

- min_samples_leaf: 1 (default)
  Minimum samples required at leaf node.

### Data Split
- test_size: 0.2 (80-20 split)
- stratify: yes (maintains genre distribution)
- random_state: 42 (reproducibility)


## OUTPUT FILES
---------------

1. trained_model.pkl
   Contains: RandomForestClassifier + LabelEncoder + metadata
   Used for: inference and model loading

2. Console Output
   Accuracy scores, classification reports, confusion matrix, feature importance


## FEATURES IMPORTANCE
----------------------

RandomForest assesses feature importance using:
- Mean Decrease in Impurity (MDI)
- How much each feature contributes to splitting decisions

Top features typically include:
- MFCC coefficients (Mel-Frequency Cepstral Coefficients)
- Spectral centroid (brightness of sound)
- Spectral features (RMS, rolloff, bandwidth)
- Chroma features (musical harmony content)


## EXAMPLES AND TESTS
--------------------

See step3_examples.py for:
- Example 1: Basic training and evaluation
- Example 2: Cross-validation analysis
- Example 3: Feature importance analysis
- Example 4: Model persistence (save/load)
- Example 5: Misclassified samples analysis
- Example 6: Probability predictions
- Example 7: Hyperparameter effects


## POTENTIAL ISSUES & SOLUTIONS
--------------------------------

Issue: Model accuracy too low (< 70%)
Solution: 
  - Check feature normalization in Step 2
  - Try increasing n_estimators (200, 300, 500)
  - Check for class imbalance in confusion matrix

Issue: Model is overfitting (train >> test accuracy)
Solution:
  - Set max_depth to limit tree complexity
  - Increase min_samples_split or min_samples_leaf
  - Use more features if available

Issue: Certain genres misclassified (high off-diagonal in confusion matrix)
Solution:
  - Check feature importance - which features distinguish genres?
  - Visualize confused genres - are they musically similar?
  - Consider genre-specific feature engineering


## NEXT STEPS
-------------

After Step 3 (this module), consider:
- Step 4: Model optimization (hyperparameter tuning, feature selection)
- Step 5: Advanced analysis (beat tracking, onset detection)
- Deployment: Flask/FastAPI REST API for real-time genre prediction
- Visualization: t-SNE or UMAP for feature space visualization


## FILES
-------

Core Components:
  model_training.py..................GenreClassificationModel
  model_evaluation.py................ModelEvaluator
  model_pipeline.py..................ClassificationPipeline

Execution:
  train_model.py.....................Main training script
  step3_examples.py..................Educational examples
  this file..........................Documentation

Expected Input:
  features_extracted.csv.............From Step 2

Expected Output:
  trained_model.pkl..................Serialized model
"""


# QUICK START
# ===========

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Quick Start: Execute 'python train_model.py' to train the model")
    print("="*80 + "\n")
