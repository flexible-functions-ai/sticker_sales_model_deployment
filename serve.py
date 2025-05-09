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

# Function to load or train a model
@app.function(image=fastai_image, volumes={"/data": data_volume})
def serve_model():
   """Load or train an XGBoost model"""
   import xgboost as xgb
   from fastai.tabular.all import add_datepart, TabularPandas, cont_cat_split
   from fastai.tabular.all import Categorify, FillMissing, Normalize, CategoryBlock, RandomSplitter, range_of
   from pathlib import Path
   import pickle
   import os
   import bentoml
   
   # Model tag used in train.py
   model_tag = "sticker_sales_v1"
   
   # Create a path to save the model for future use
   model_path = "/data/sticker_sales_model.pkl"
   
   try:
       # First attempt: Try loading from BentoML
       print(f"Attempting to load model from BentoML with tag '{model_tag}'...")
       try:
           bento_model = bentoml.xgboost.load_model(model_tag)
           print(f"Successfully loaded model from BentoML.")
           return bento_model
       except Exception as e:
           print(f"Could not load from BentoML: {str(e)}")
       
       # Second attempt: Try loading from pickle
       if os.path.exists(model_path):
           print(f"Loading existing model from pickle at {model_path}")
           with open(model_path, 'rb') as f:
               model = pickle.load(f)
           return model
       
       # Third attempt: Train a new model if neither option worked
       print("No existing model found. Training new model...")
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
       
       # Save model to both formats
       
       # 1. Save to BentoML
       print(f"Saving model to BentoML with tag '{model_tag}'...")
       bentoml.xgboost.save_model(
           model_tag, 
           xgb_model,
           custom_objects={
               "preprocessor": {
                   "cont_names": cont_names,
                   "cat_names": cat_names
               }
           }
       )
       
       # 2. Save to pickle
       print(f"Saving model to pickle at {model_path}")
       with open(model_path, 'wb') as f:
           pickle.dump(xgb_model, f)
       
       # Save preprocessing info separately
       preproc_path = "/data/sticker_sales_preproc.pkl"
       print(f"Saving preprocessing info to {preproc_path}...")
       preproc_info = {
           "cont_names": cont_names,
           "cat_names": cat_names,
           "procs": [Categorify, FillMissing, Normalize]
       }
       with open(preproc_path, 'wb') as f:
           pickle.dump(preproc_info, f)
       
       # Ensure changes are committed to the volume
       volume.commit()
       
       print("Model training and saving complete!")
       return xgb_model
       
   except Exception as e:
       import traceback
       print(f"Error loading/training model: {str(e)}")
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
       # First, load or train model
       model = serve_model.remote()
       
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
   
   # Pre-load the model to ensure it exists
   print("Preparing model...")
   serve_model.remote()
   print("Model preparation complete!")
   
   print("\nAPI is ready for use at:")
   print("- Health check: https://flexible-functions-ai--sticker-sales-api-health.modal.run")
   print("- CSV predictions: https://flexible-functions-ai--sticker-sales-api-predict-csv.modal.run")