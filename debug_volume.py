import modal
from pathlib import Path

app = modal.App("debug-volume")
volume = modal.Volume.from_name("sticker-data-volume")

@app.function(volumes={"/data": volume})
def check_volume_contents():
    """Debug function to see what's actually in the volume"""
    import os
    
    print("🔍 Checking volume contents...")
    print("=" * 50)
    
    # List all files in /data
    data_path = Path("/data")
    if data_path.exists():
        print(f"📁 Contents of /data:")
        for item in data_path.rglob("*"):
            if item.is_file():
                print(f"   📄 {item}")
            elif item.is_dir():
                print(f"   📁 {item}/")
    else:
        print("❌ /data directory doesn't exist!")
    
    # Specifically check for feast_utils.py
    feast_utils_path = Path("/data/feast_utils.py")
    if feast_utils_path.exists():
        print(f"✅ feast_utils.py found at {feast_utils_path}")
        print(f"📊 File size: {feast_utils_path.stat().st_size} bytes")
    else:
        print("❌ feast_utils.py NOT found in /data")
    
    # Check feature_repo
    feature_repo_path = Path("/data/feature_repo")
    if feature_repo_path.exists():
        print(f"✅ feature_repo found at {feature_repo_path}")
        for item in feature_repo_path.rglob("*"):
            print(f"   📄 {item}")
    else:
        print("❌ feature_repo NOT found in /data")

@app.local_entrypoint()
def main():
    check_volume_contents.remote()