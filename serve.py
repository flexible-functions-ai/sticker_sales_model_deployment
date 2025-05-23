import modal
import pandas as pd
import numpy as np
from fastapi import File, UploadFile, Form, HTTPException
import io
import sys

# Create app definition
app = modal.App("sticker-sales-api")

# Image with Feast and serving dependencies
feast_image = modal.Image.debian_slim().pip_install([
    "feast>=0.34.0",           # Feast feature store
    "fastapi==0.95.2",         # Web framework
    "uvicorn==0.22.0",         # ASGI server
    "bentoml==1.3.2",          # Model serving
    "xgboost==1.7.6",          # Gradient boosting
    "scikit-learn==1.3.1",     # ML utilities
    "pandas",                  # Data manipulation
    "numpy",                   # Numerical computing
    "fastai",                  # For date features
    "torch",                   # PyTorch (for FastAI)
    "pyarrow"                  # Parquet support
])

# Create volume to access data
data_volume = modal.Volume.from_name("sticker-data-volume")

# Simple health endpoint
@app.function(image=feast_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint to verify the API is running"""
    return {
        "status": "healthy", 
        "service": "sticker-sales-feast-api",
        "feature_store": "feast_enabled"
    }

# Function to load Feast-enabled model
@app.function(image=feast_image, volumes={"/data": data_volume})
def serve_feast_model():
    """
    Load Feast-enabled XGBoost model.
    
    This function:
    1. Tries to load model from BentoML
    2. Falls back to pickle file
    3. Trains new model if none exists
    """
    import xgboost as xgb
    from pathlib import Path
    import pickle
    import os
    import bentoml
    
    # Add the data directory to Python path
    sys.path.append('/data')
    
    print("🤖 Loading Feast-enabled model...")
    
    try:
        from feast_utils import FeastFeatureProcessor
    except ImportError:
        print("❌ Error: feast_utils.py not found in volume")
        raise
    
    # Model paths
    model_tag = "sticker_sales_feast_v1"
    model_path = "/data/sticker_sales_feast_model.pkl"
    
    try:
        # Try loading from BentoML first
        print(f"🔍 Attempting to load Feast model from BentoML with tag '{model_tag}'...")
        try:
            bento_model = bentoml.xgboost.load_model(model_tag)
            print(f"✅ Successfully loaded Feast model from BentoML.")
            return bento_model
        except Exception as e:
            print(f"⚠️ Could not load from BentoML: {str(e)}")
        
        # Try loading from pickle
        if os.path.exists(model_path):
            print(f"💾 Loading existing Feast model from pickle at {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                print(f"📊 Model info: {model_data.get('feature_count', 'unknown')} features, RMSE: {model_data.get('training_rmse', 'unknown')}")
                return model_data['model']
            else:
                return model_data
        
        # If no model exists, train a new one
        print("🔧 No existing Feast model found. Training new model...")
        
        # Initialize Feast processor
        processor = FeastFeatureProcessor(
            repo_path="/data/feature_repo",
            data_path="/data"
        )
        
        # Load and process training data
        path = Path('/data/')
        train_df = pd.read_csv(