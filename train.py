import modal
import pandas as pd
import numpy as np
from fastai.tabular.all import *  # Move this import to the top level
import xgboost as xgb
import bentoml
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
    
    print(f"Model saved: {model_tag}")
    return str(model_tag)

@app.local_entrypoint()
def main():
    # Train the model remotely
    print("Starting model training on Modal...")
    model_tag = train_model.remote()
    print(f"Model training completed. Model tag: {model_tag}")