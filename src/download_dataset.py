import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("cicdataset/cicids2017")

print("Path to dataset files:", path)

# Copy files to our data directory if needed, or just use the path
target_dir = r"c:\Users\ekonk\OneDrive\Desktop\vibe\researchCyber\paper1\data"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Dataset successfully copied to {target_dir}")
