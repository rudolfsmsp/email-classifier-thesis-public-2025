import pandas as pd
import os
import json

data_dir = "data"

files = {
    "utwente": os.path.join(data_dir, "utwente/phishing_validation_emails.csv"),
    "sandhya": os.path.join(data_dir, "sandhya_devpriya_emails/dataset_5971.csv"),
    "oibsip_spam": os.path.join(data_dir, "oibsip_spam/spam.csv"),
    "wiechmann": os.path.join(data_dir, "wiechmann_emails/enron_spam_data.csv"),
    "suhasmaddali": os.path.join(data_dir, "suhasmaddali_emails/train.csv"),
    "nahmias": os.path.join(data_dir, "nahmiasd_emails")
}

# loads a csv file and returns a dataframe
def load_csv(dataset_name, file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] file not found -> {file_path}")
        return pd.DataFrame()
    print(f"[INFO] processing dataset from '{dataset_name}'...")
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8", encoding_errors="ignore")
    except Exception as e:
        print(f"[ERROR] failed processing {file_path}: {e}")
        return pd.DataFrame()
    df.columns = [col.lower() for col in df.columns]
    print(f"[SUCCESS] finished processing dataset from '{dataset_name}'. Rows -> {len(df)}")
    return df

# loads all json files recursively from the nahmias dataset
def load_nahmias_json(directory):
    print(f"[INFO] processing nahmias dataset from {directory}...")
    data = []
    if os.path.exists(directory):
        for root, dirs, files_list in os.walk(directory):
            for file_name in files_list:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            email_data = json.load(f)
                        email_text = email_data.get("email_subject", "") + " " + email_data.get("email_body", "")
                        data.append({"email_text": email_text, "email_type": "safe email"})
                    except Exception as e:
                        print(f"[ERROR] failed to read {file_path}: {e}")
    else:
        print(f"[ERROR] directory not found: {directory}")
    df = pd.DataFrame(data)
    print(f"[SUCCESS] finished processing nahmias dataset. Rows -> {len(df)}")
    return df

# processes all datasets and creates the unified dataset
def main():
    print("[INFO] starting unified dataset generation...")
    utwente_df = load_csv("utwente", files["utwente"])
    sandhya_df = load_csv("sandhya", files["sandhya"])
    oibsip_spam_df = load_csv("oibsip_spam", files["oibsip_spam"])
    wiechmann_df = load_csv("wiechmann", files["wiechmann"])
    suhasmaddali_df = load_csv("suhasmaddali", files["suhasmaddali"])
    nahmias_df = load_nahmias_json(files["nahmias"])
    print("[INFO] combining all processed datasets...")
    unified_df = pd.concat(
        [utwente_df, sandhya_df, oibsip_spam_df, wiechmann_df, suhasmaddali_df, nahmias_df],
        ignore_index=True
    )
    output_file = "unified_email_dataset.csv"
    unified_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] finished dataset processing. saved unified dataset -> {output_file}")
    print("[INFO] first few rows:")
    print(unified_df.head())

if __name__ == "__main__":
    main()
