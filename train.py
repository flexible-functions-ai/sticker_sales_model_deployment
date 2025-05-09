import modal
import pandas as pd
import numpy as np
from fastai.tabular.all import *  # Move this import to the top level
import xgboost as xgb
import bentoml
import pickle
from pathlib import Path
import os

# Define Modal resources
app = modal.App("sticker-sales-forecast")
image = modal.Image.debian_slim().pip_install([
    "fastai", 
    "xgboost", 
    "bentoml", 
    "scikit-learn", 
    "pandas", 
    "numpy", 
    "torch"
])
volume = modal.Volume.from_name("sticker-data-volume")

# Define paths for pickle-based model saving
MODEL_PATH = "/data/sticker_sales_model.pkl"
PREPROC_PATH = "/data/sticker_sales_preproc.pkl"

@app.function(image=image, volumes={"/data": volume})
def train_model():
    # No need to import fastai.tabular.all here since we moved it to the top
    
    # Set up paths
    path = Path('/data/')
    
    # Check if data files exist
    print("Files available in volume:")
    for file in path.glob("*"):
        print(f" - {file}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(path/'train.csv', index_col='id')
    test_df = pd.read_csv(path/'test.csv', index_col='id')
    
    # Data preprocessing
    print("Preprocessing data...")
    train_df = train_df.dropna(subset=['num_sold'])
    train_df = add_datepart(train_df, 'date', drop=False)
    test_df = add_datepart(test_df, 'date', drop=False)
    
    # Feature preparation
    cont_names, cat_names = cont_cat_split(train_df, dep_var='num_sold')
    splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
    to = TabularPandas(train_df, procs=[Categorify, FillMissing, Normalize],
                      cat_names=cat_names,
                      cont_names=cont_names,
                      y_names='num_sold',
                      y_block=CategoryBlock(),
                      splits=splits)
    dls = to.dataloaders(bs=64)
    
    # Prepare training data
    X_train, y_train = to.train.xs, to.train.ys.values.ravel()
    X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor()
    xgb_model = xgb_model.fit(X_train, y_train)
    
    # Save model with BentoML
    print("Saving model with BentoML...")
    model_tag = bentoml.xgboost.save_model(
        "sticker_sales_v1", 
        xgb_model,
        custom_objects={
            "preprocessor": {
                "cont_names": cont_names,
                "cat_names": cat_names
            }
        }
    )
    
    # Save model with pickle
    print(f"Saving model with pickle to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save preprocessing info separately
    print(f"Saving preprocessing info to {PREPROC_PATH}...")
    preproc_info = {
        "cont_names": cont_names,
        "cat_names": cat_names,
        "procs": [Categorify, FillMissing, Normalize]
    }
    with open(PREPROC_PATH, 'wb') as f:
        pickle.dump(preproc_info, f)
    
    # Ensure changes are committed to the volume
    volume.commit()
    
    print(f"Model saved: {model_tag} and to pickle files")
    return str(model_tag)

@app.local_entrypoint()
def main():
    # Train the model remotely
    print("Starting model training on Modal...")
    model_tag = train_model.remote()
    print(f"Model training completed. Model tag: {model_tag}")
    print(f"Model and preprocessing info also saved as pickle files at {MODEL_PATH} and {PREPROC_PATH}")