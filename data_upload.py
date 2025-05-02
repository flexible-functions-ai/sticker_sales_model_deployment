import modal
import sys
from pathlib import Path

# Create an app for the data upload (using App instead of Stub)
app = modal.App("sticker-data-upload")

# Create a volume to persist data
volume = modal.Volume.from_name("sticker-data-volume", create_if_missing=True)

@app.function(volumes={"/data": volume})
def upload_data(local_data_path):
    import shutil
    import os
    
    # Ensure the destination directory exists
    os.makedirs("/data", exist_ok=True)
    
    # Copy all files from the local data directory to the volume
    for file in Path(local_data_path).glob("*"):
        dest = f"/data/{file.name}"
        if file.is_file():
            shutil.copy(file, dest)
            print(f"Copied {file} to {dest}")
    
    # List files to confirm upload
    print("\nFiles in Modal volume:")
    for file in Path("/data").glob("*"):
        print(f" - {file}")

@app.local_entrypoint()
def main():
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "./data"  # Default path
    
    print(f"Uploading data from {data_path}")
    upload_data.remote(data_path)