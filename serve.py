import modal
import pandas as pd
import numpy as np
from fastapi import File, UploadFile, Form, HTTPException
import io

# Create app definition
app = modal.App("sticker-sales-api")

# Define base image with all dependencies
base_image = (modal.Image.debian_slim()
         .pip_install("pydantic==1.10.8")        
         .pip_install("fastapi==0.95.2")         
         .pip_install("uvicorn==0.22.0")         
         .pip_install("bentoml==1.3.2")          
         .pip_install([                         
             "xgboost==1.7.6",
             "scikit-learn==1.3.1",
             "pandas",
             "numpy",
         ]))

# Create the fastai image by extending the base image
fastai_image = (base_image
                .pip_install(["fastai", "torch"]))

# Create volume to access data
data_volume = modal.Volume.from_name("sticker-data-volume")

# Simple health endpoint
@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint to verify the API is running"""
    return {"status": "healthy", "service": "sticker-sales-api"}

# Function to train and save model - this is a direct function without any BentoML
@app.function(image=fastai_image, volumes={"/data": data_volume})
def train_xgboost_model():
    """Train an XGBoost model and return it directly"""
    import xgboost as xgb
    from fastai.tabular.all import add_datepart, TabularPandas, cont_cat_split
    from fastai.tabular.all import Categorify, FillMissing, Normalize, CategoryBlock, RandomSplitter, range_of
    from pathlib import Path
    import pickle
    import os
    
    # Create a path to save the model for future use
    model_path = "/data/sticker_sales_model.pkl"
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    try:
        print("Training new model...")
        # Load and preprocess training data
        path = Path('/data/')
        
        print("Loading training data...")
        train_df = pd.read_csv(path/'train.csv', index_col='id')
        
        # Drop rows with missing target values
        train_df = train_df.dropna(subset=['num_sold'])
        
        # Add date features
        print("Preprocessing data...")
        train_df = add_datepart(train_df, 'date', drop=False)
        
        # Feature preparation
        cont_names, cat_names = cont_cat_split(train_df, dep_var='num_sold')
        splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
        
        # Create TabularPandas processor
        to = TabularPandas(train_df, 
                          procs=[Categorify, FillMissing, Normalize],
                          cat_names=cat_names,
                          cont_names=cont_names,
                          y_names='num_sold',
                          y_block=CategoryBlock(),
                          splits=splits)
        
        # Prepare training data
        X_train, y_train = to.train.xs, to.train.ys.values.ravel()
        
        # Train a simple XGBoost model
        print("Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(n_estimators=100)
        xgb_model.fit(X_train, y_train)
        
        # Save model to file
        print(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        print("Model training complete!")
        return xgb_model
        
    except Exception as e:
        import traceback
        print(f"Error training model: {str(e)}")
        print(traceback.format_exc())
        raise

# CSV upload endpoint 
@app.function(image=fastai_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="POST")
async def predict_csv(file: UploadFile = File(...)):
    """API endpoint for batch predictions from a CSV file"""
    import xgboost as xgb
    import io
    import pickle
    from fastai.tabular.all import add_datepart, TabularPandas, cont_cat_split
    from fastai.tabular.all import Categorify, FillMissing, Normalize, CategoryBlock, RandomSplitter, range_of
    from pathlib import Path
    
    try:
        # First, train or load model
        model = train_xgboost_model.remote()
        
        # Read uploaded CSV file content
        contents = await file.read()
        
        # Parse CSV data
        try:
            test_df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse uploaded CSV: {str(e)}"
            }
        
        # Load the training data for preprocessing
        path = Path('/data/')
        train_df = pd.read_csv(path/'train.csv', index_col='id')
        train_df = train_df.dropna(subset=['num_sold'])
        
        # Add date features to both datasets
        train_df = add_datepart(train_df, 'date', drop=False)
        test_df = add_datepart(test_df, 'date', drop=False)
        
        # Feature preparation
        cont_names, cat_names = cont_cat_split(train_df, dep_var='num_sold')
        splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
        
        # Create TabularPandas processor
        to = TabularPandas(train_df, 
                          procs=[Categorify, FillMissing, Normalize],
                          cat_names=cat_names,
                          cont_names=cont_names,
                          y_names='num_sold',
                          y_block=CategoryBlock(),
                          splits=splits)
        
        # Create a test dataloader
        dls = to.dataloaders(bs=64)
        test_dl = dls.test_dl(test_df)
        
        # Make predictions using our model
        predictions = model.predict(test_dl.xs)
        
        # Return the predictions as a simple list
        return predictions.tolist()
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing CSV: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing the API"""
    print("Starting sticker-sales-api...")
    
    # Pre-train the model to ensure it exists
    print("Training model...")
    train_xgboost_model.remote()
    print("Model training complete!")
    
    print("\nAPI is ready for use at:")
    print("- Health check: https://flexible-functions-ai--sticker-sales-api-health.modal.run")
    print("- CSV predictions: https://flexible-functions-ai--sticker-sales-api-predict-csv.modal.run")