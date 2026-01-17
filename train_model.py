"""
Heart Disease Prediction - Model Training Script
=================================================

This script follows the exact steps from the Jupyter Notebook:
1. Load and explore the dataset
2. Remove outliers using Z-score method
3. Train multiple models (Decision Tree, Random Forest, AdaBoost)
4. Select the best model
5. Save the trained model for deployment

Author: Generated for Heart Disease Prediction Project
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

# ============================================================================
# STEP 1: LOAD THE DATASET
# ============================================================================
print("=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)

# Load the heart disease dataset
df = pd.read_csv('heart.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"\nDataset Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nTarget Distribution:")
print(df['target'].value_counts())

# ============================================================================
# STEP 2: DATA PREPROCESSING - OUTLIER REMOVAL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Removing Outliers using Z-Score Method")
print("=" * 70)

# Calculate Z-scores for all columns
z_scores = np.abs(stats.zscore(df))

# Keep only rows where all Z-scores are less than 3
# This removes outliers that are more than 3 standard deviations away
data_clean = df[(z_scores < 3).all(axis=1)]

print(f"\nOriginal Dataset Shape: {df.shape}")
print(f"Cleaned Dataset Shape: {data_clean.shape}")
print(f"Rows Removed: {df.shape[0] - data_clean.shape[0]}")

# ============================================================================
# STEP 3: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Preparing Data for Training")
print("=" * 70)

# Separate features (X) and target (y)
X = data_clean.drop('target', axis=1)
y = data_clean['target']

print(f"\nFeature Columns: {list(X.columns)}")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Samples: {X.shape[0]}")

# Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=0
)

print(f"\nTraining Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 4: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training Multiple Models")
print("=" * 70)

models = {}
results = {}

# ----------------------------------------------------------------------------
# Model 1: Decision Tree
# ----------------------------------------------------------------------------
print("\n--- Training Decision Tree Classifier ---")
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)

models['Decision Tree'] = dtree
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt),
    'roc_auc': roc_auc_score(y_test, y_pred_dt)
}

print(f"Accuracy: {results['Decision Tree']['accuracy']*100:.2f}%")
print(f"F1-Score: {results['Decision Tree']['f1']:.4f}")
print(f"Precision: {results['Decision Tree']['precision']:.4f}")
print(f"Recall: {results['Decision Tree']['recall']:.4f}")

# ----------------------------------------------------------------------------
# Model 2: Random Forest
# ----------------------------------------------------------------------------
print("\n--- Training Random Forest Classifier ---")
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred_rf = rfc.predict(X_test)

models['Random Forest'] = rfc
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_rf)
}

print(f"Accuracy: {results['Random Forest']['accuracy']*100:.2f}%")
print(f"F1-Score: {results['Random Forest']['f1']:.4f}")
print(f"Precision: {results['Random Forest']['precision']:.4f}")
print(f"Recall: {results['Random Forest']['recall']:.4f}")

# ----------------------------------------------------------------------------
# Model 3: AdaBoost
# ----------------------------------------------------------------------------
print("\n--- Training AdaBoost Classifier ---")
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

models['AdaBoost'] = ada
results['AdaBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_ada),
    'precision': precision_score(y_test, y_pred_ada),
    'recall': recall_score(y_test, y_pred_ada),
    'f1': f1_score(y_test, y_pred_ada),
    'roc_auc': roc_auc_score(y_test, y_pred_ada)
}

print(f"Accuracy: {results['AdaBoost']['accuracy']*100:.2f}%")
print(f"F1-Score: {results['AdaBoost']['f1']:.4f}")
print(f"Precision: {results['AdaBoost']['precision']:.4f}")
print(f"Recall: {results['AdaBoost']['recall']:.4f}")

# ============================================================================
# STEP 5: SELECT BEST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Model Comparison and Selection")
print("=" * 70)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values('accuracy', ascending=False)

print("\nModel Performance Comparison:")
print(comparison_df.round(4))

# Select best model based on accuracy (primary) and F1-score (secondary)
best_model_name = comparison_df.index[0]
best_model = models[best_model_name]

print(f"\n[SUCCESS] Best Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

# Print detailed classification report for best model
print(f"\nDetailed Classification Report for {best_model_name}:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# ============================================================================
# STEP 6: SAVE THE MODEL AND METADATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Saving Model and Metadata")
print("=" * 70)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the best model
model_path = 'models/heart_disease_model.pkl'
joblib.dump(best_model, model_path)
print(f"[SUCCESS] Model saved to: {model_path}")

# Save model metadata (feature names, model name, performance metrics)
metadata = {
    'model_name': best_model_name,
    'feature_names': list(X.columns),
    'accuracy': float(results[best_model_name]['accuracy']),
    'f1_score': float(results[best_model_name]['f1']),
    'precision': float(results[best_model_name]['precision']),
    'recall': float(results[best_model_name]['recall']),
    'roc_auc': float(results[best_model_name]['roc_auc']),
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0])
}

import json
metadata_path = 'models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"[SUCCESS] Metadata saved to: {metadata_path}")

# Save all model results for reference
results_path = 'models/all_models_results.json'
all_results = {}
for model_name, metrics in results.items():
    all_results[model_name] = {k: float(v) for k, v in metrics.items()}

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"[SUCCESS] All model results saved to: {results_path}")

print("\n" + "=" * 70)
print("[SUCCESS] TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Model File: {model_path}")
print(f"\nYou can now use this model in the Streamlit app!")
