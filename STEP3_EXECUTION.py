"""
STEP 3: MACHINE LEARNING MODEL TRAINING - EXECUTION GUIDE

Trains Random Forest model for genre classification using features 
extracted in Step 2.

QUICK START (for the impatient):
================================
    python train_model.py

This will:
1. Load features from Step 2
2. Train RandomForest classifier  
3. Print accuracy, metrics, feature importance
4. Save model to trained_model.pkl


DETAILED WORKFLOW:
==================

STEP 3A: VERIFY ALL DEPENDENCIES ARE INSTALLED
───────────────────────────────────────────────
pip install -r requirements.txt

Should include:
  - scikit-learn >= 1.3.0 (for RandomForest, metrics)
  - numpy >= 1.24.0
  - pandas >= 2.0.0
  - joblib >= 1.3.0  (for model serialization)


STEP 3B: ENSURE FEATURES ARE EXTRACTED (STEP 2 COMPLETE)
─────────────────────────────────────────────────────────
Check if this file exists:
  data/features_extracted.csv

If not, run Step 2 first:
  python step2_pipeline.py

Expected format:
  - 1000 audio files
  - 82+ features (normalized)
  - Genre labels


STEP 3C: CHOOSE YOUR EXECUTION MODE
─────────────────────────────────────

MODE 1: FULL TRAINING (Recommended for first run)
──────────────────────────────────────────────────
    python train_model.py

Output:
  [1/4] Preparing data...
    ✓ Training set: 800 samples
    ✓ Test set: 200 samples
    
  [2/4] Training model...
    ✓ RandomForestClassifier training...
    ✓ Model training complete
    
  [3/4] Evaluating model...
    ✓ Accuracy: 0.75
    ✓ Classification report with precision, recall, f1
    ✓ Confusion matrix
    
  [4/4] Computing feature importance...
    ✓ Top important features listed
    
  ✓ Model saved to trained_model.pkl


MODE 2: EDUCATIONAL EXAMPLES (Learn how to use the modules)
────────────────────────────────────────────────────────────
    python step3_examples.py

Includes 7 examples:
  1. Basic Training and Evaluation
  2. Cross-Validation Analysis (5-fold)
  3. Feature Importance Analysis
  4. Model Persistence (Save/Load)
  5. Misclassified Samples Analysis
  6. Probability Predictions
  7. Hyperparameter Effects (comparing n_estimators)


MODE 3: CUSTOM TRAINING (Advanced: modify parameters)
──────────────────────────────────────────────────────
Edit and run this Python code:

    from model_pipeline import ClassificationPipeline
    import pandas as pd
    
    # Load features
    df_features = pd.read_csv("data/features_extracted.csv")
    
    # Create pipeline
    pipeline = ClassificationPipeline(random_state=42)
    
    # Train with custom parameters
    results = pipeline.run_full_pipeline(
        df_features,
        test_size=0.2,           # 80-20 split
        n_estimators=200,        # More trees = potentially better (but slower)
        max_depth=30,            # Limit tree depth to reduce overfitting
        save_model=True,         # Save for future inference
        model_path="my_model.pkl"
    )
    
    # Access results
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")


STEP 3D: USING THE TRAINED MODEL FOR INFERENCE
────────────────────────────────────────────────

Loading and predicting with saved model:

    from model_training import GenreClassificationModel
    import numpy as np
    
    # Load model
    model = GenreClassificationModel()
    model.load_model("trained_model.pkl")
    
    # Make predictions on new data
    predictions = model.predict(X_test)          # Class indices
    probabilities = model.predict_proba(X_test)  # Softmax probabilities
    
    # Get class labels for predictions
    predicted_genres = model.class_labels[predictions]
    
    # Get feature importance
    importance = model.get_feature_importance(top_n=20)


UNDERSTANDING THE OUTPUT
========================

MODEL EVALUATION REPORT includes:

1. ACCURACY METRICS
   Accuracy: Overall correct predictions
   - Range: 0.0 to 1.0
   - 0.8 = 80% of classifications correct

2. PER-CLASS METRICS (for each of 10 genres)
   
   Precision: Of predicted class X, what % are correct?
     - High precision = few false positives
     - Formula: TP / (TP + FP)
   
   Recall: Of actual class X, what % did we find?
     - High recall = few false negatives
     - Formula: TP / (TP + FN)
   
   F1-Score: Harmonic mean of precision and recall
     - Balanced metric
     - Formula: 2 * (P * R) / (P + R)

3. CONFUSION MATRIX
   Shows which genres are confused with which:
   
         Predicted
         blues classical country disco
    Actual blues  50     2       1      0
         classical 1   48      0      2
         country   0     0     47      3
         disco     1     1      2     46
   
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications

4. FEATURE IMPORTANCE (Top 20)
   Which features matter most for classification?
   
   Most important features typically:
   - MFCC coefficients
   - Spectral centroid
   - Spectral features


HYPERPARAMETER TUNING
===================

Common parameters to tune:

n_estimators (default: 100)
  Increase to → More accurate but slower
  Decrease to → Faster but less accurate
  Range: 50-500
  
  Effect: Higher usually better until convergence
  Example runs: 50, 100, 200, 300

max_depth (default: None)
  Increase to → More complex trees, risk overfitting
  Decrease to → Simpler trees, may underfit
  Range: 5-50
  
  Effect: Reduces overfitting
  Typical: 20-30

min_samples_split (default: 2)
  Increase to → Fewer splits, smoother boundaries
  Typical: 2-10
  
min_samples_leaf (default: 1)
  Increase to → Smoother predictions
  Typical: 1-5


DEBUGGING COMMON ISSUES
======================

Issue: Low Accuracy (< 70%)
└─ Check: Are features properly normalized? (Step 2)
└─ Try: Increase n_estimators (200, 300)
└─ Try: Remove max_depth limit


Issue: Different accuracy each run (Not reproducible)
└─ Fix: Ensure random_state=42 is set everywhere
└─ Check: model_training.py has random_state=42 in:
          - train_test_split()
          - RandomForestClassifier()


Issue: High train accuracy but low test accuracy (Overfitting)
└─ Problem: Model memorized training data
└─ Fix: Decrease max_depth
└─ Fix: Increase min_samples_split
└─ Fix: Increase min_samples_leaf


Issue: Certain genres always misclassified
└─ Check: Confusion matrix to see which ones
└─ Analyze: Feature importance - do these genres have similar features?
└─ Consider: Are these genres musically similar? (e.g., blues-rock)


FILES INVOLVED IN STEP 3
=======================

CORE ML MODULES:
  model_training.py ........... GenreClassificationModel class
  model_evaluation.py ......... ModelEvaluator class for metrics
  model_pipeline.py .......... ClassificationPipeline orchestration

EXECUTION SCRIPTS:
  train_model.py .............. Main training entry point
  step3_examples.py ........... 7 educational examples

DOCUMENTATION:
  this file ................... Execution guide
  STEP3_GUIDE.md ............. Technical documentation

INPUT:
  data/features_extracted.csv . From Step 2 (1000×82 features)

OUTPUT:
  trained_model.pkl .......... Serialized model + encoder
  Console output ............. Accuracy, metrics, confusion matrix


WHAT HAPPENS DURING TRAINING
============================

1. PREPARE DATA (30 seconds)
   ├─ Load features_extracted.csv
   ├─ Extract feature columns (82 features)
   ├─ Extract genre labels
   ├─ Encode genres: blues=0, classical=1, ..., reggae=9
   ├─ Stratified split: 800 train / 200 test
   │  (maintains class distribution)
   └─ Return X_train, X_test, y_train, y_test, LabelEncoder

2. TRAIN MODEL (1-2 seconds)
   ├─ Create RandomForestClassifier (100 trees)
   ├─ Fit on training data
   ├─ Each tree learns different genres
   └─ Trees vote on predictions (ensemble)

3. EVALUATE MODEL (1 second)
   ├─ Predict on training set
   ├─ Predict on test set
   ├─ Compute accuracy
   ├─ Compute precision, recall, f1 per class
   ├─ Build confusion matrix
   └─ Print comprehensive report

4. GET FEATURE IMPORTANCE (1 second)
   ├─ Each feature importance = how much splits use it
   ├─ Normalize by total importance
   ├─ Sort by importance
   └─ Display top 20

5. SAVE MODEL (< 1 second)
   ├─ Serialize RandomForestClassifier
   ├─ Serialize LabelEncoder
   ├─ Save feature column names
   ├─ Package metadata
   └─ Write to trained_model.pkl using joblib


EXPECTED ACCURACY RANGES
=========================

Based on genre classification tasks:
  Baseline (random): 10% (1/10 classes)
  Poor model: 30-50%
  Okay model: 60-70%
  Good model: 75-85%
  Great model: 85%+

Our typical results: ~75-80% accuracy


NEXT STEPS AFTER STEP 3
=======================

After training and evaluating the model, you can:

1. Optimize Performance
   ├─ Tune hyperparameters
   ├─ Feature selection
   ├─ Try different algorithms (SVM, XGBoost)
   └─ Cross-validation

2. Deploy Model
   ├─ Flask/FastAPI REST API
   ├─ Real-time genre prediction
   └─ Batch processing

3. Advanced Analysis
   ├─ Feature visualization (t-SNE, PCA)
   ├─ Per-genre analysis
   └─ Misclassification patterns

4. Extended Features
   ├─ Beat tracking (Step 5)
   ├─ Onset detection
   └─ Rhythm patterns


REFERENCE
=========

Scikit-learn RandomForest:
https://scikit-learn.org/stable/modules/ensemble.html#forests

Classification Metrics Guide:
https://scikit-learn.org/stable/modules/model_evaluation.html

Model Persistence (joblib):
https://scikit-learn.org/stable/modules/model_persistence.html
"""


# QUICK EXECUTION EXAMPLES
# ========================

if __name__ == "__main__":
    import sys
    
    print(__doc__)
    print("\n" + "="*80)
    print("READY TO START?")
    print("="*80)
    print("Option 1 - Full Training (recommended):")
    print("  python train_model.py")
    print("\nOption 2 - Educational Examples:")
    print("  python step3_examples.py")
    print("="*80 + "\n")
