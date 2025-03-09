import pandas as pd
import os

DATA_DIR = "data"
FILES = {
    "tarun": os.path.join(DATA_DIR, "tarun_phishing_urls/phishing_site_urls.csv"),
    "phiusiil": os.path.join(DATA_DIR, "phiusill_phishing_urls/PhiUSIIL_Phishing_URL_Dataset.csv")
}

# processes phishing url datasets from given file paths and returns a concatenated dataframe
def clean_phishing_url_data(files):
    print("[INFO] starting phishing URL dataset creation...")
    dfs = []
    for source, file_path in files.items():
        if os.path.exists(file_path):
            print(f"[INFO] processing dataset from '{source}'...")
            try:
                df = pd.read_csv(file_path)
                df.columns = [col.lower() for col in df.columns]
                dfs.append(df)
                print(f"[SUCCESS] finished processing dataset from '{source}'. Rows: {len(df)}")
            except Exception as e:
                print(f"[ERROR] finished processing dataset from '{source}' with failure: {e}")
        else:
            print(f"[ERROR] file not found: {file_path}")
    print("[SUCCESS] finished phishing URL dataset creation.")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# executes dataset processing and saves unified url dataset
def main():
    print("[INFO] starting unified URL dataset generation...")
    phishing_urls_df = clean_phishing_url_data({"tarun": FILES["tarun"], "phiusiil": FILES["phiusiil"]})
    output_file = "unified_url_dataset.csv"
    phishing_urls_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] finished unified URL dataset generation. Saved -> {output_file}")

if __name__ == "__main__":
    main()
