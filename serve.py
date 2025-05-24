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

# Image with Feast and serving dependencies - AGGRESSIVE pandas fix
feast_image = (modal.Image.debian_slim()
               .pip_install("setuptools>=68.0.0")
               .pip_install([
                   "fastapi==0.95.2",         # Web framework
                   "uvicorn==0.22.0",         # ASGI server
                   "bentoml==1.3.2",          # Model serving
                   "xgboost==1.7.6",          # Gradient boosting
                   "scikit-learn==1.3.1",     # ML utilities
                   "numpy",                   # Numerical computing
                   "torch",                   # PyTorch (install before fastai)
                   "pyarrow"                  # Parquet support
               ])
               .pip_install("pandas>=2.0.0")  # Install pandas after torch but before fastai
               .pip_install("fastai")          # Install fastai after pandas
               .pip_install("feast>=0.34.0")   # Install feast last
               .pip_install("pandas>=2.0.0", force_build=True))  # Force upgrade pandas again

# Create volume to access data
data_volume = modal.Volume.from_name("sticker-data-volume")

# Enhanced health endpoint with version debugging
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint with version debugging"""
    import os
    from pathlib import Path
    
    # Check volume contents and versions
    volume_info = {
        "status": "healthy",
        "service": "sticker-sales-feast-api",
        "feature_store": "feast_enabled",
        "volume_contents": {},
        "debug_info": {},
        "versions": {}
    }
    
    try:
        # Check versions
        volume_info["versions"]["pandas"] = pd.__version__
        try:
            import dask
            volume_info["versions"]["dask"] = dask.__version__
        except ImportError:
            volume_info["versions"]["dask"] = "not_installed"
        
        try:
            import feast
            volume_info["versions"]["feast"] = feast.__version__
        except ImportError:
            volume_info["versions"]["feast"] = "not_installed"
        
        # Check /data directory
        data_path = Path("/data")
        if data_path.exists():
            volume_info["volume_contents"]["data_exists"] = True
            volume_info["volume_contents"]["data_files"] = [str(f) for f in data_path.glob("*")]
        else:
            volume_info["volume_contents"]["data_exists"] = False
        
        # Check feast_utils.py specifically
        feast_utils_path = Path("/data/feast_utils.py")
        volume_info["debug_info"]["feast_utils_exists"] = feast_utils_path.exists()
        if feast_utils_path.exists():
            volume_info["debug_info"]["feast_utils_size"] = feast_utils_path.stat().st_size
        
        # Check feature_repo
        feature_repo_path = Path("/data/feature_repo")
        volume_info["debug_info"]["feature_repo_exists"] = feature_repo_path.exists()
        if feature_repo_path.exists():
            volume_info["debug_info"]["feature_repo_contents"] = [str(f) for f in feature_repo_path.glob("*")]
        
    except Exception as e:
        volume_info["debug_info"]["error"] = str(e)
    
    return volume_info

# Test pandas version endpoint
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="GET")
def test_pandas():
    """Test pandas version and Dask compatibility"""
    import pandas as pd
    
    result = {
        "pandas_version": pd.__version__,
        "pandas_version_info": pd.__version__.split('.'),
        "test_results": {}
    }
    
    # Test if pandas version is >= 2.0.0
    major_version = int(pd.__version__.split('.')[0])
    result["test_results"]["pandas_major_version_ge_2"] = major_version >= 2
    
    # Test Dask import
    try:
        import dask
        result["dask_version"] = dask.__version__
        result["test_results"]["dask_import_success"] = True
    except Exception as e:
        result["test_results"]["dask_import_success"] = False
        result["test_results"]["dask_import_error"] = str(e)
    
    # Test feast_utils import
    sys.path.insert(0, '/data')
    try:
        import feast_utils
        result["test_results"]["feast_utils_import_success"] = True
    except Exception as e:
        result["test_results"]["feast_utils_import_success"] = False
        result["test_results"]["feast_utils_import_error"] = str(e)
    
    return result

# Simplified serving function for testing
@app.function(image=feast_image, volumes={"/data": data_volume})
def serve_feast_model():
    """Simplified model serving for testing"""
    print(f"🔍 DEBUG: Pandas version: {pd.__version__}")
    
    # Add /data to Python path
    sys.path.insert(0, '/data')
    
    try:
        from feast_utils import FeastFeatureProcessor
        print("✅ Successfully imported FeastFeatureProcessor")
        return "mock_model"  # Return a mock model for now
    except Exception as e:
        print(f"❌ Error importing FeastFeatureProcessor: {e}")
        raise

# Simplified CSV prediction endpoint
@app.function(image=feast_image, volumes={"/data": data_volume})
@modal.fastapi_endpoint(method="POST")
async def predict_csv_feast(file: UploadFile = File(...)):
    """Simplified API endpoint for testing pandas/dask compatibility"""
    
    print("🚀 Starting Feast-enabled prediction...")
    print(f"🔍 Pandas version: {pd.__version__}")
    
    # Add /data to Python path
    sys.path.insert(0, '/data')
    
    try:
        # Test Dask import first
        try:
            import dask
            print(f"✅ Dask version: {dask.__version__}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Dask import failed: {str(e)}",
                "pandas_version": pd.__version__
            }
        
        # Test feast_utils import
        try:
            from feast_utils import FeastFeatureProcessor
            print("✅ Successfully imported FeastFeatureProcessor")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import feast_utils: {str(e)}",
                "pandas_version": pd.__version__
            }
        
        # Read uploaded CSV file content
        contents = await file.read()
        test_df = pd.read_csv(io.BytesIO(contents))
        
        # For now, return mock predictions to test the pipeline
        predictions = [100.0 + i * 10 for i in range(len(test_df))]
        
        return {
            "success": True,
            "predictions": predictions,
            "model_info": {
                "pandas_version": pd.__version__,
                "feature_count": "mock_test",
                "prediction_count": len(predictions),
                "feature_store": "feast_test_mode"
            }
        }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing CSV: {str(e)}",
            "traceback": traceback.format_exc(),
            "pandas_version": pd.__version__
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing the Feast API"""
    print("🚀 Starting Feast-enabled sticker-sales-api...")
    print("✅ API deployment complete!")
    
    print("\n🌐 Test endpoints:")
    print("- Health check: https://flexible-functions-ai--sticker-sales-api-health.modal.run")
    print("- Pandas test: https://flexible-functions-ai--sticker-sales-api-test-pandas.modal.run")
    print("- CSV predictions: https://flexible-functions-ai--sticker-sales-api-predict-csv-feast.modal.run")