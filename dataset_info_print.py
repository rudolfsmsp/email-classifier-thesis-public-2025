import pandas as pd
import os

DATA_DIR = "data"
FILES = {
    "tarun": os.path.join(DATA_DIR, "tarun_phishing_urls/phishing_site_urls.csv"),
    "phiusiil": os.path.join(DATA_DIR, "phiusill_phishing_urls/PhiUSIIL_Phishing_URL_Dataset.csv")
}

# prints dataset summary information for the given dataframe and dataset name
def dataset_info_print(df, dataset_name):
    print(f"[INFO] === {dataset_name} dataset overview ===")
    print(f"[INFO] total entries: {len(df)}")
    if 'email_type' in df.columns:
        print("[INFO] email type distribution:")
        print(df['email_type'].value_counts())
    if 'label' in df.columns:
        print("[INFO] url label distribution:")
        print(df['label'].value_counts())
    print("[INFO] duplicate rows:")
    print(df.duplicated().sum())

# processes dataset files and prints dataset information
def main():
    print("[INFO] starting dataset information processing...")
    datasets = {
        "unified email dataset": "unified_email_dataset.csv",
        "master email dataset": "master_email_dataset.csv",
        "unified url dataset": "unified_url_dataset.csv",
        "master url dataset": "master_url_dataset.csv"
    }
    for dataset_name, file_path in datasets.items():
        if not os.path.exists(file_path):
            print(f"[ERROR] {dataset_name}: file not found -> {file_path}")
            continue
        try:
            print(f"[INFO] processing {dataset_name}...")
            df = pd.read_csv(file_path, low_memory=False)
            dataset_info_print(df, dataset_name)
            print(f"[SUCCESS] finished processing {dataset_name}.")
        except Exception as e:
            print(f"[ERROR] {dataset_name}: error reading {file_path} -> {e}")
    print("[INFO] finished dataset information processing.")

if __name__ == "__main__":
    main()
