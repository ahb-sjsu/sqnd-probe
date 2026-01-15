# @title 10. Save & Download Results { display-mode: "form" }
# @markdown Persist results to Google Drive and optionally download as zip

import shutil

print("=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Always persist results to Drive
if SAVE_DIR and os.path.exists(SAVE_DIR):
    print(f"\nPersisting to: {SAVE_DIR}")

    # Save final results JSON
    if os.path.exists("results/final_results.json"):
        dest = f"{SAVE_DIR}/final_results.json"
        shutil.copy("results/final_results.json", dest)
        print(f"  Saved: final_results.json")

    # Save splits config
    if os.path.exists("data/splits/all_splits.json"):
        dest = f"{SAVE_DIR}/all_splits.json"
        shutil.copy("data/splits/all_splits.json", dest)
        print(f"  Saved: all_splits.json")

    # Models are already saved to SAVE_DIR during training
    model_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pt")]
    if model_files:
        print(f"  Models already in Drive: {len(model_files)} files")
        for mf in model_files[:5]:
            print(f"    - {mf}")
        if len(model_files) > 5:
            print(f"    ... and {len(model_files)-5} more")

    print(f"\nResults persisted to Google Drive: {SAVE_DIR}")
else:
    print("WARNING: SAVE_DIR not available, results only in local directories")

# Optional: Create download zip
if CREATE_DOWNLOAD_ZIP:
    import zipfile

    zip_path = "BIP_v10.10_results.zip"
    print(f"\n" + "-" * 60)
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

    print(f"Download package ready: {zip_path}")

    # Download in Colab, or show path otherwise
    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        print(f"Not running in Colab. Zip saved to: {os.path.abspath(zip_path)}")
else:
    print(f"\n(Zip download disabled - set CREATE_DOWNLOAD_ZIP=True in cell 1 to enable)")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
