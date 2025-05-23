import modal
import sys
from pathlib import Path

# Create an app for the data upload
app = modal.App("sticker-data-upload")

# Image with Feast dependencies - FIXED to include setuptools
image = modal.Image.debian_slim().pip_install([
    "setuptools",              # FIXED: provides distutils for Python 3.12
    "feast>=0.34.0",           # Feast feature store
    "fastai",                  # For add_datepart function
    "pandas",                  # Data manipulation
    "pyarrow",                 # Parquet file support
    "scikit-learn"             # ML utilities
])

# Create a volume to persist data
volume = modal.Volume.from_name("sticker-data-volume", create_if_missing=True)

@app.function(image=image, volumes={"/data": volume})
def upload_data_and_init_feast(local_data_path):
    """
    Upload data to Modal volume and initialize Feast feature store.
    
    This function:
    1. Copies all data files to Modal volume
    2. Copies Feast configuration and utilities
    3. Initializes Feast feature registry
    4. Prepares and materializes features if training data exists
    """
    import shutil
    import os
    import pandas as pd
    import subprocess
    
    print("🚀 Starting data upload and Feast initialization...")
    
    # Ensure the destination directory exists
    os.makedirs("/data", exist_ok=True)
    
    # Copy all files from the local data directory to the volume
    print("📁 Copying data files...")
    for file in Path(local_data_path).glob("*"):
        dest = f"/data/{file.name}"
        if file.is_file():
            shutil.copy(file, dest)
            print(f"   ✅ Copied {file} to {dest}")
    
    # Copy feature repository to volume
    feature_repo_path = Path("feature_repo")
    if feature_repo_path.exists():
        dest_repo = "/data/feature_repo"
        if os.path.exists(dest_repo):
            shutil.rmtree(dest_repo)
        shutil.copytree(feature_repo_path, dest_repo)
        print(f"   ✅ Copied feature_repo to {dest_repo}")
    else:
        print("   ❌ feature_repo directory not found!")
        return
    
    # Copy feast utilities (from root level)
    feast_utils_path = Path("feast_utils.py")
    if feast_utils_path.exists():
        shutil.copy(feast_utils_path, "/data/feast_utils.py")
        print("   ✅ Copied feast_utils.py to volume")
    else:
        print("   ❌ feast_utils.py not found in root directory!")
        return
    
    # Initialize Feast feature store
    print("🎛️ Initializing Feast feature store...")
    os.chdir("/data")
    
    try:
        # Apply feature definitions to Feast
        print("   📝 Applying feature definitions...")
        result = subprocess.run(
            ["feast", "apply"], 
            cwd="/data/feature_repo",
            capture_output=True, 
            text=True,
            check=True
        )
        print("   ✅ Feast apply successful:")
        print(f"   📄 {result.stdout}")
        
        # Create and materialize features if we have training data
        if os.path.exists("/data/train.csv"):
            print("🔧 Preparing and materializing features...")
            
            # Import utilities here after they're available
            sys.path.append('/data')
            from feast_utils import FeastFeatureProcessor
            
            # Load training data
            print("   📊 Loading training data...")
            train_df = pd.read_csv("/data/train.csv")
            print(f"   📈 Training data shape: {train_df.shape}")
            
            # Process features
            processor = FeastFeatureProcessor(
                repo_path="/data/feature_repo", 
                data_path="/data"
            )
            feature_df = processor.prepare_feature_data(train_df, is_training=True)
            processor.save_feature_data(feature_df)
            
            # Materialize features to SQLite online store
            print("   🏪 Materializing features to SQLite online store...")
            result = subprocess.run(
                ["feast", "materialize-incremental", "2017-01-01T00:00:00"],
                cwd="/data/feature_repo",
                capture_output=True,
                text=True,
                check=True
            )
            print("   ✅ Feature materialization successful:")
            print(f"   📄 {result.stdout}")
        else:
            print("   ⚠️ No train.csv found, skipping feature materialization")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Feast command failed: {e}")
        print(f"📄 Stdout: {e.stdout}")
        print(f"📄 Stderr: {e.stderr}")
        raise
    
    # List files to confirm upload
    print("\n📋 Files in Modal volume:")
    for file in Path("/data").glob("*"):
        print(f" - {file}")
    
    print("🎉 Data upload and Feast initialization complete!")

@app.local_entrypoint()
def main():
    """
    Local entry point for data upload.
    """
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "./data"  # Default path
    
    print(f"🎯 Uploading data from {data_path} and initializing Feast...")
    upload_data_and_init_feast.remote(data_path)