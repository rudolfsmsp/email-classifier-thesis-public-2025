import pandas as pd
import os

DATA_DIR = "data"
DATASET_PATH = "master_url_dataset.csv"
USER_PROVIDED_PATH = "user_provided_urls.csv"

FILES = {
    "tarun": os.path.join(DATA_DIR, "tarun_phishing_urls/phishing_site_urls.csv"),
    "phiusiil": os.path.join(DATA_DIR, "phiusill_phishing_urls/PhiUSIIL_Phishing_URL_Dataset.csv")
}

# processes phishing url datasets from given files and returns a concatenated dataframe
def clean_phishing_url_data(files):
    print("[INFO] starting phishing URL dataset creation...")
    dfs = []
    for source, file_path in files.items():
        if os.path.exists(file_path):
            print(f"[INFO] processing dataset from '{source}'...")
            try:
                df = pd.read_csv(file_path, dtype=str, low_memory=False)
                df.columns = [col.lower() for col in df.columns]
                required_columns = ["url", "label"]
                df = df[[col for col in required_columns if col in df.columns]]
                dfs.append(df)
                print(f"[SUCCESS] finished processing dataset from '{source}'. Rows -> {len(df)}")
            except Exception as e:
                print(f"[ERROR] finished processing dataset from '{source}' with failure: {e}")
        else:
            print(f"[ERROR] file not found: {file_path}")
    if not dfs:
        print("[ERROR] finished phishing URL dataset creation with no data found.")
        return pd.DataFrame(columns=["url", "label"])
    print("[SUCCESS] finished phishing URL dataset creation successfully.")
    return pd.concat(dfs, ignore_index=True)

# loads user provided urls from csv file if available
def load_user_provided_data():
    if os.path.exists(USER_PROVIDED_PATH):
        print("[INFO] merging user-provided URL dataset...")
        try:
            user_df = pd.read_csv(USER_PROVIDED_PATH, dtype=str, names=["url", "label"])
            print(f"[SUCCESS] loaded {len(user_df)} user-provided URLs.")
            return user_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided URLs: {e}")
            return pd.DataFrame(columns=["url", "label"])
    return pd.DataFrame(columns=["url", "label"])

# creates master url dataset from unified dataset and user provided data using label mapping
def create_master_url_dataset():
    print("[INFO] starting master URL dataset creation...")
    dataset_path = "unified_url_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] {dataset_path} not found. Cannot create master URL dataset.")
        return
    try:
        df = pd.read_csv(dataset_path, dtype=str, low_memory=False)
    except Exception as e:
        print(f"[ERROR] finished master URL dataset creation with failure: {e}")
        return
    df.columns = [col.lower() for col in df.columns]
    df = df.drop_duplicates(subset=["url"])
    label_map = {
        "bad": "2",
        "safe": "0",
        "good": "0",
        "0": "0",
        "1": "2"
    }
    df["label"] = df["label"].map(lambda x: label_map.get(x.lower().strip(), x))
    df.dropna(subset=["url", "label"], inplace=True)
    user_df = load_user_provided_data()
    df = pd.concat([df, user_df], ignore_index=True).drop_duplicates(subset=["url"])
    df.to_csv(DATASET_PATH, index=False)
    print(f"[SUCCESS] finished master URL dataset creation. Saved -> {DATASET_PATH}")
    print(f"[SUCCESS] loaded {len(df)} urls from master_url_dataset.csv")

# executes the dataset processing pipeline
def main():
    print("[INFO] starting dataset processing pipeline...")
    phishing_urls_df = clean_phishing_url_data(FILES)
    output_file = "unified_url_dataset.csv"
    phishing_urls_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] finished unified URL dataset creation. Saved -> {output_file}")
    if os.path.exists(output_file):
        create_master_url_dataset()
    else:
        print("[ERROR] unified_url_dataset.csv does not exist. Cannot create master URL dataset.")
    print("[INFO] finished dataset processing pipeline.")

if __name__ == "__main__":
    main()
