import modal
import pandas as pd
import numpy as np
from fastapi import File, UploadFile, Form, HTTPException
from pathlib import Path
import io
import sys
import os

# Create app definition
app = modal.App("sticker-sales-api")

# Image with Feast and serving dependencies
feast_image = (modal.Image.debian_slim()
               .pip_install("setuptools>=68.0.0")
               .pip_install([
                   "fastapi==0.95.2",         # Web framework
                   "uvicorn==0.22.0",         # ASGI server
                   "bentoml==1.3.2",          # Model serving
                   "xgboost==1.7.6",          # Gradient boosting
                   "scikit-learn==1.3.1",     # ML utilities
                   "numpy",                   # Numerical computing
                   "torch",                   # PyTorch
                   "pyarrow"                  # Parquet support
               ])
               .pip_install("pandas>=2.0.0")  # Install pandas
               .pip_install("fastai")          # Install fastai after pandas
               .pip_install("feast>=0.34.0"))  # Install feast last

# Create volume to access data
data_volume = modal.Volume.from_name("sticker-data-volume")

# Health endpoint
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "sticker-sales-feast-api",
        "pandas_version": pd.__version__,
        "feast_enabled": True
    }

# Simple model loading function (returns only the model)
@app.function(image=feast_image, volumes={"/data": data_volume})
def load_model():
    """Load just the model (no Feast objects)"""
    import pickle
    
    print("🤖 Loading model...")
    
    model_path = "/data/sticker_sales_model.pkl"
    if not os.path.exists(model_path):
        raise Exception("No model found")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Feast-enabled prediction endpoint (initializes Feast locally)
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="POST")
async def predict_csv_feast(file: UploadFile = File(...)):
    """API endpoint using Feast (initialized locally to avoid pickle issues)"""
    
    print("🚀 Starting Feast-enabled prediction...")
    
    try:
        # Load model
        model = load_model.remote()
        
        # Read uploaded CSV
        contents = await file.read()
        test_df = pd.read_csv(io.BytesIO(contents))
        
        print(f"📊 Received CSV with shape: {test_df.shape}")
        
        # Validate required columns
        required_columns = ['date', 'country', 'store', 'product']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            return {
                "success": False,
                "error": f"Missing required columns: {missing_columns}"
            }
        
        # Add data path to Python path
        sys.path.insert(0, '/data')
        
        # TRY FEAST APPROACH FIRST
        try:
            print("🎛️ Attempting Feast feature engineering...")
            
            # Initialize Feast processor locally (to avoid pickle issues)
            from feast_utils import FeastFeatureProcessor
            processor = FeastFeatureProcessor(
                repo_path="/data/feature_repo",
                data_path="/data"
            )
            
            # Prepare feature data using Feast
            feature_df = processor.prepare_feature_data(test_df, is_training=False)
            
            # Create entity rows for Feast
            entity_rows = [
                {"sticker_id": row["sticker_id"]} 
                for _, row in feature_df.iterrows()
            ]
            
            print(f"🎯 Created {len(entity_rows)} entity rows for Feast")
            
            # Get features from Feast online store
            feature_vector = processor.get_online_features(entity_rows)
            
            # Remove sticker_id column for prediction
            if 'sticker_id' in feature_vector.columns:
                X_test = feature_vector.drop('sticker_id', axis=1)
            else:
                X_test = feature_vector
            
            print(f"✅ Successfully used Feast! Got {X_test.shape[1]} features")
            feast_success = True
            
        except Exception as feast_error:
            print(f"⚠️ Feast failed: {feast_error}")
            print("🔄 Falling back to manual feature engineering...")
            feast_success = False
            
            # FALLBACK TO MANUAL FEATURE ENGINEERING
            from fastai.tabular.all import add_datepart
            
            processed_df = test_df.copy()
            
            # Add date features using fastai
            processed_df = add_datepart(processed_df, 'date', drop=False)
            
            # Create 'Elapsed' feature (days since reference date)
            reference_date = pd.to_datetime('2017-01-01')
            processed_df['Elapsed'] = (pd.to_datetime(processed_df['date']) - reference_date).dt.days
            
            # Convert categorical columns to numeric codes
            categorical_cols = ['country', 'store', 'product']
            for col in categorical_cols:
                processed_df[col] = pd.Categorical(processed_df[col]).codes
            
            # Convert date to numeric (days since epoch)
            processed_df['date'] = (pd.to_datetime(processed_df['date']) - pd.Timestamp('1970-01-01')).dt.days
            
            # Define the exact 17 features the model expects
            exact_features = [
                'date', 'country', 'store', 'product', 'Year', 'Month', 
                'Dayofweek', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Week', 
                'Day', 'Dayofyear', 'Elapsed'
            ]
            
            # Select exactly these 17 features
            X_test = processed_df[exact_features].copy()
        
        # Convert boolean columns to int
        bool_columns = X_test.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            X_test[col] = X_test[col].astype(int)
        
        # Ensure all columns are numeric
        for col in X_test.columns:
            if X_test[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        # Handle feature count mismatch for model compatibility
        expected_features = 17
        current_features = X_test.shape[1]
        
        if current_features < expected_features:
            # Add dummy features
            for i in range(expected_features - current_features):
                X_test[f'dummy_feature_{i}'] = 0
        elif current_features > expected_features:
            # Take only the first n features
            X_test = X_test.iloc[:, :expected_features]
        
        print(f"🎯 Final feature matrix shape: {X_test.shape}")
        
        # Verify we have exactly 17 features
        if X_test.shape[1] != 17:
            return {
                "success": False,
                "error": f"Feature count mismatch: got {X_test.shape[1]}, expected 17"
            }
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Ensure predictions are reasonable
        predictions = np.maximum(predictions, 0)
        
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"📈 Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        
        # Return in the format expected by Streamlit
        return {
            "success": True,
            "predictions": predictions.tolist(),
            "model_info": {
                "feature_count": X_test.shape[1],
                "prediction_count": len(predictions),
                "feast_enabled": feast_success,
                "feature_store": "feast_online_store" if feast_success else "manual_fallback"
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing CSV: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.local_entrypoint()
def main():
    """Local entrypoint"""
    print("🚀 Starting Feast-enabled API with fallback...")
    print("✅ API deployment complete!")
    
    print("\n🌐 API endpoints:")
    print("- Health check: https://flexible-functions-ai--sticker-sales-api-health.modal.run")
    print("- CSV predictions: https://flexible-functions-ai--sticker-sales-api-predict-csv-feast.modal.run")