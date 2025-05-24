import modal
from pathlib import Path

app = modal.App("direct-upload")
volume = modal.Volume.from_name("sticker-data-volume")

@app.local_entrypoint()
def main():
    """Upload files directly using Modal's volume upload"""
    print("🚀 Starting direct file upload...")
    
    # Upload feast_utils.py
    if Path("feast_utils.py").exists():
        print("📁 Uploading feast_utils.py...")
        with volume.batch_upload() as batch:
            batch.put_file("feast_utils.py", "/feast_utils.py")
        print("✅ Uploaded feast_utils.py")
    else:
        print("❌ feast_utils.py not found locally")
    
    # Upload feature_repo directory
    if Path("feature_repo").exists():
        print("📁 Uploading feature_repo directory...")
        with volume.batch_upload() as batch:
            batch.put_directory("feature_repo/", "/feature_repo/")
        print("✅ Uploaded feature_repo directory")
    else:
        print("❌ feature_repo directory not found locally")
    
    print("🎉 Upload complete!")