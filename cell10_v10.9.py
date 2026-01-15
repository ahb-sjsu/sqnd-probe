# @title 10. Download Results { display-mode: "form" }
# @markdown Download all models and results

import zipfile

zip_path = "BIP_v10.9_results.zip"
print("Creating download package...")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    # Results
    if os.path.exists("results/final_results.json"):
        zf.write("results/final_results.json")

    # Models (from Drive)
    if SAVE_DIR and os.path.exists(SAVE_DIR):
        for f in os.listdir(SAVE_DIR):
            if f.endswith(".pt"):
                zf.write(f"{SAVE_DIR}/{f}", f"models/{f}")

    # Config
    if os.path.exists("data/splits/all_splits.json"):
        zf.write("data/splits/all_splits.json")

print(f"\nDownload package ready: {zip_path}")

# Download in Colab, or show path otherwise
try:
    from google.colab import files

    files.download(zip_path)
except ImportError:
    print(f"Not running in Colab. Results saved to: {os.path.abspath(zip_path)}")
