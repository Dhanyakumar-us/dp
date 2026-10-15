import os
import sys
import tarfile
import urllib.request
import time

dest_dir = "D:/cifar_data"
os.makedirs(dest_dir, exist_ok=True)
dest_path = os.path.join(dest_dir, "cifar-10-python.tar.gz")
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    total_size = int(resp.headers.get("Content-Length", 0))

print(f"Total size: {total_size / (1024*1024):.1f} MB, already downloaded: {existing_size / (1024*1024):.1f} MB")

if existing_size < total_size:
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    downloaded = existing_size
    
    mode = "ab" if existing_size > 0 else "wb"
    with urllib.request.urlopen(req) as resp, open(dest_path, mode) as f:
        while True:
            chunk = resp.read(1024 * 1024) # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = (downloaded - existing_size) / (elapsed + 1e-5) / (1024 * 1024)
            pct = downloaded / total_size * 100
            print(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) - Speed: {speed:.2f} MB/s", end="", flush=True)

print("\nDownload complete! Verifying and extracting...")
with tarfile.open(dest_path, "r:gz") as tar:
    tar.extractall(path=dest_dir)
print("Extracted successfully to:", dest_dir)
