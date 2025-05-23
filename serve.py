import modal
import pandas as pd
import numpy as np
from fastapi import File, UploadFile, Form, HTTPException
import io
import sys

# Create app definition
app = modal.App("sticker-sales-api")

# Image with Feast and serving dependencies - FIXED to include setuptools
feast_image = modal.Image.debian_slim().pip_install([
    "setuptools",              # FIXED: provides distutils for Python 3.12
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
        train_df = pd.read_csv(path/'train.csv', index_col='id')
        train_df = train_df.dropna(subset=['num_sold'])
        
        print("🔧 Processing training data with Feast...")
        
        # Prepare and save feature data
        feature_df = processor.prepare_feature_data(train_df, is_training=True)
        processor.save_feature_data(feature_df)
        
        # Get training features
        entity_df = feature_df[['sticker_id', 'event_timestamp']].copy()
        training_features = processor.get_training_features(entity_df)
        
        # Merge with targets
        train_with_id = train_df.reset_index()
        train_with_id['sticker_id'] = (
            train_with_id['country'].astype(str) + "_" + 
            train_with_id['store'].astype(str) + "_" + 
            train_with_id['product'].astype(str)
        )
        
        final_training_df = training_features.merge(
            train_with_id[['sticker_id', 'num_sold']],
            on='sticker_id',
            how='inner'
        )
        
        # Prepare training data
        feature_columns = [col for col in final_training_df.columns 
                          if col not in ['sticker_id', 'event_timestamp', 'num_sold']]
        
        X = final_training_df[feature_columns]
        y = final_training_df['num_sold']
        
        # Train model
        print("🤖 Training XGBoost model with Feast features...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        
        # Save model
        model_data = {
            'model': xgb_model,
            'feature_columns': feature_columns,
            'feast_repo_path': "/data/feature_repo",
            'model_type': "feast_enabled",
            'feature_count': len(feature_columns)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save to BentoML
        bentoml.xgboost.save_model(
            model_tag,
            xgb_model,
            custom_objects={
                "feature_columns": feature_columns,
                "feast_repo_path": "/data/feature_repo",
                "model_type": "feast_enabled",
                "feature_count": len(feature_columns)
            }
        )
        
        data_volume.commit()
        
        print("🎉 Feast model training and saving complete!")
        return xgb_model
        
    except Exception as e:
        import traceback
        print(f"❌ Error loading/training Feast model: {str(e)}")
        print(traceback.format_exc())
        raise

# CSV upload endpoint with Feast
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="POST")
async def predict_csv_feast(file: UploadFile = File(...)):
    """
    API endpoint for batch predictions using Feast features.
    
    This endpoint:
    1. Loads the trained model
    2. Processes uploaded CSV with Feast
    3. Gets features from Feast online store
    4. Makes predictions
    5. Returns predictions as JSON
    """
    import xgboost as xgb
    import io
    import pickle
    from pathlib import Path
    
    # Add the data directory to Python path
    sys.path.append('/data')
    
    print("🚀 Starting Feast-enabled prediction...")
    
    try:
        from feast_utils import FeastFeatureProcessor
    except ImportError:
        return {
            "success": False,
            "error": "Feast utilities not available. Please ensure feast_utils.py is uploaded."
        }
    
    try:
        # Load the trained Feast model
        print("🤖 Loading trained model...")
        model = serve_feast_model.remote()
        
        # Read uploaded CSV file content
        contents = await file.read()
        
        # Parse CSV data
        try:
            test_df = pd.read_csv(io.BytesIO(contents))
            print(f"📊 Received CSV with shape: {test_df.shape}")
            print(f"📝 Columns: {list(test_df.columns)}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse uploaded CSV: {str(e)}"
            }
        
        # Validate required columns
        required_columns = ['date', 'country', 'store', 'product']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            return {
                "success": False,
                "error": f"Missing required columns: {missing_columns}. Required: {required_columns}"
            }
        
        # Initialize Feast processor
        print("🎛️ Initializing Feast processor...")
        processor = FeastFeatureProcessor(
            repo_path="/data/feature_repo",
            data_path="/data"
        )
        
        # Prepare test data for Feast
        print("🔧 Preparing test data for Feast...")
        test_feature_df = processor.prepare_feature_data(test_df, is_training=False)
        
        # Create entity rows for online feature retrieval
        entity_rows = [
            {"sticker_id": row["sticker_id"]} 
            for _, row in test_feature_df.iterrows()
        ]
        
        print(f"🎯 Created {len(entity_rows)} entity rows for feature retrieval")
        
        # Get online features from Feast
        print("⚡ Retrieving features from Feast online store...")
        feature_vector = processor.get_online_features(entity_rows)
        
        # Load model metadata to get expected feature columns
        model_path = "/data/sticker_sales_feast_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            expected_columns = model_data.get('feature_columns', [])
            print(f"📋 Expected feature columns: {expected_columns}")
        else:
            # Fallback: use all non-entity columns
            expected_columns = [col for col in feature_vector.columns 
                              if col != 'sticker_id']
            print(f"⚠️ Using fallback feature columns: {expected_columns}")
        
        # Check if we have all expected features
        available_columns = [col for col in expected_columns if col in feature_vector.columns]
        missing_features = [col for col in expected_columns if col not in feature_vector.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            # Fill missing features with default values
            for col in missing_features:
                feature_vector[col] = 0
            
        # Select and order features
        X_test = feature_vector[expected_columns]
        
        print(f"🎯 Making predictions with feature matrix shape: {X_test.shape}")
        print(f"📊 Feature matrix columns: {list(X_test.columns)}")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"📈 Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        
        # Return predictions
        return {
            "success": True,
            "predictions": predictions.tolist(),
            "model_info": {
                "feature_count": len(expected_columns),
                "prediction_count": len(predictions),
                "feature_store": "feast_enabled"
            }
        }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing CSV with Feast: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing the Feast API"""
    print("🚀 Starting Feast-enabled sticker-sales-api...")
    
    # Pre-load the model to ensure it exists
    print("🔧 Preparing Feast model...")
    serve_feast_model.remote()
    print("✅ Feast model preparation complete!")
    
    print("\n🌐 Feast API is ready for use at:")
    print("- Health check: https://flexible-functions-ai--sticker-sales-api-health.modal.run")
    print("- CSV predictions: https://flexible-functions-ai--sticker-sales-api-predict-csv-feast.modal.run")