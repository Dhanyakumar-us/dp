import os
import urllib.request
import time

DATA_DIR = "D:/cifar_data"
os.makedirs(DATA_DIR, exist_ok=True)

urls = {
    "train.parquet": "https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/train/0.parquet",
    "test.parquet": "https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/test/0.parquet",
}

headers = {"User-Agent": "Mozilla/5.0"}

for fname, url in urls.items():
    dest = os.path.join(DATA_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
        print(f"{fname} already exists ({os.path.getsize(dest)/(1024*1024):.1f} MB), skipping.")
        continue
    
    print(f"Downloading {fname} from Hugging Face CDN...")
    t0 = time.time()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(2 * 1024 * 1024) # 2MB chunk
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            speed = downloaded / (time.time() - t0 + 1e-5) / (1024 * 1024)
            pct = downloaded / total * 100 if total else 0
            print(f"\r{fname}: {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct:.1f}%) - {speed:.2f} MB/s", end="", flush=True)
    print(f"\n{fname} downloaded successfully in {time.time() - t0:.1f}s.")

print("All CIFAR-10 data ready on D:/cifar_data!")
