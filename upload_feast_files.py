import modal
import sys
from pathlib import Path
import os

app = modal.App("upload-feast-files")

image = (modal.Image.debian_slim()
         .pip_install("setuptools>=68.0.0")
         .pip_install(["pandas", "pathlib"]))

volume = modal.Volume.from_name("sticker-data-volume", create_if_missing=True)

@app.function(image=image, volumes={"/data": volume})
def upload_feast_files():
    """Upload feast_utils.py and feature_repo specifically"""
    import shutil
    import os
    
    print("🚀 Uploading Feast files...")
    
    # Check what we have in the current working directory in the container
    print(f"📁 Container working directory: {os.getcwd()}")
    print(f"📁 Container contents: {os.listdir('.')}")
    
    # The files should be mounted from your local directory
    # Let's check what's available
    local_files = []
    for item in Path('.').glob('*'):
        local_files.append(str(item))
        print(f"   Found: {item}")
    
    # Try to copy feast_utils.py if it exists
    if Path('feast_utils.py').exists():
        shutil.copy('feast_utils.py', '/data/feast_utils.py')
        print("✅ Copied feast_utils.py to volume")
    else:
        print("❌ feast_utils.py not found in mounted directory")
    
    # Try to copy feature_repo if it exists
    if Path('feature_repo').exists():
        if os.path.exists('/data/feature_repo'):
            shutil.rmtree('/data/feature_repo')
        shutil.copytree('feature_repo', '/data/feature_repo')
        print("✅ Copied feature_repo to volume")
        
        # List contents
        for item in Path('/data/feature_repo').rglob('*'):
            print(f"   Copied: {item}")
    else:
        print("❌ feature_repo not found in mounted directory")
    
    # Commit to volume
    volume.commit()
    
    # Verify what's now in the volume
    print("\n📋 Final volume contents:")
    for item in Path('/data').glob('*'):
        print(f"   {item}")
    
    print("🎉 Upload complete!")

@app.local_entrypoint()
def main():
    upload_feast_files.remote()