import modal
import pandas as pd
import numpy as np
import xgboost as xgb
import bentoml
import pickle
from pathlib import Path
import os
import sys

# Define Modal resources
app = modal.App("sticker-sales-forecast")

# Image with Feast and ML dependencies - FIXED to include setuptools
image = modal.Image.debian_slim().pip_install([
    "setuptools",              # FIXED: provides distutils for Python 3.12
    "feast>=0.34.0",           # Feast feature store
    "fastai",                  # For date features
    "xgboost",                 # Gradient boosting
    "bentoml",                 # Model packaging
    "scikit-learn",            # ML utilities
    "pandas",                  # Data manipulation
    "numpy",                   # Numerical computing
    "torch",                   # PyTorch (for FastAI)
    "pyarrow"                  # Parquet support
])

volume = modal.Volume.from_name("sticker-data-volume")

@app.function(image=image, volumes={"/data": volume})
def train_model():
    """
    Train XGBoost model using Feast features.
    
    This function:
    1. Loads training data
    2. Uses Feast to generate consistent features
    3. Trains XGBoost model
    4. Saves model with BentoML and pickle
    """
    # Add the data directory to Python path
    sys.path.append('/data')
    
    print("🚀 Starting Feast-enabled model training...")
    
    # Import Feast utilities
    try:
        from feast_utils import FeastFeatureProcessor
        from feast import FeatureStore
    except ImportError as e:
        print(f"❌ Error importing Feast utilities: {e}")
        print("💡 Make sure feast_utils.py and feature_repo are uploaded")
        raise
    
    # Set up paths
    path = Path('/data/')
    
    print("📋 Files available in volume:")
    for file in path.glob("*"):
        print(f" - {file}")
    
    # Load data
    print("📊 Loading training data...")
    train_df = pd.read_csv(path/'train.csv', index_col='id')
    
    # Remove rows with missing target values
    print(f"📈 Original training data shape: {train_df.shape}")
    train_df = train_df.dropna(subset=['num_sold'])
    print(f"📉 After removing missing targets: {train_df.shape}")
    
    # Initialize Feast processor
    print("🎛️ Initializing Feast feature processor...")
    processor = FeastFeatureProcessor(
        repo_path="/data/feature_repo",
        data_path="/data"
    )
    
    # Prepare feature data for Feast
    print("🔧 Preparing feature data...")
    feature_df = processor.prepare_feature_data(train_df, is_training=True)
    processor.save_feature_data(feature_df)
    
    # Create entity DataFrame for feature retrieval
    entity_df = feature_df[['sticker_id', 'event_timestamp']].copy()
    print(f"🎯 Entity DataFrame shape: {entity_df.shape}")
    
    # Get features using Feast
    print("🔍 Retrieving features from Feast...")
    training_features = processor.get_training_features(entity_df)
    
    # Merge with target variable
    # First, create a mapping from original index to sticker_id
    train_with_id = train_df.reset_index()
    train_with_id['sticker_id'] = (
        train_with_id['country'].astype(str) + "_" + 
        train_with_id['store'].astype(str) + "_" + 
        train_with_id['product'].astype(str)
    )
    
    # Merge training features with targets
    final_training_df = training_features.merge(
        train_with_id[['sticker_id', 'num_sold']],
        on='sticker_id',
        how='inner'
    )
    
    print(f"🎯 Final training data shape: {final_training_df.shape}")
    
    # Prepare features and target
    feature_columns = [col for col in final_training_df.columns 
                      if col not in ['sticker_id', 'event_timestamp', 'num_sold']]
    
    X = final_training_df[feature_columns]
    y = final_training_df['num_sold']
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"📝 Feature columns: {feature_columns}")
    
    # Train XGBoost model
    print("🤖 Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X, y)
    
    # Calculate training metrics
    train_predictions = xgb_model.predict(X)
    train_rmse = np.sqrt(np.mean((y - train_predictions) ** 2))
    print(f"📈 Training RMSE: {train_rmse:.4f}")
    
    # Save model with BentoML including Feast metadata
    print("💾 Saving model with BentoML...")
    model_tag = bentoml.xgboost.save_model(
        "sticker_sales_feast_v1", 
        xgb_model,
        custom_objects={
            "feature_columns": feature_columns,
            "training_rmse": train_rmse,
            "feast_repo_path": "/data/feature_repo",
            "model_type": "feast_enabled",
            "feature_count": len(feature_columns)
        }
    )
    
    # Also save as pickle for backup
    model_path = "/data/sticker_sales_feast_model.pkl"
    print(f"💾 Saving model to pickle at {model_path}...")
    
    model_data = {
        'model': xgb_model,
        'feature_columns': feature_columns,
        'training_rmse': train_rmse,
        'feast_repo_path': "/data/feature_repo",
        'model_type': "feast_enabled",
        'feature_count': len(feature_columns)
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Ensure changes are committed to the volume
    volume.commit()
    
    print(f"🎉 Model saved with Feast: {model_tag}")
    print("✅ Training completed successfully!")
    return str(model_tag)

@app.local_entrypoint()
def main():
    """
    Local entry point for model training.
    """
    print("🚀 Starting Feast-enabled model training on Modal...")
    model_tag = train_model.remote()
    print(f"🎉 Model training completed. Model tag: {model_tag}")