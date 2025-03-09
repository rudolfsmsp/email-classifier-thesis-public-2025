import pandas as pd
import os
import re

DATASET_PATH = "master_email_dataset.csv"
USER_PROVIDED_PATH = "user_provided_emails.csv"

# removes html tags, punctuation, and extra spaces from text
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# merges multiple columns into one by keeping the first non-empty value
def unify_columns(df, source_cols, final_col):
    existing = [c for c in source_cols if c in df.columns]
    if not existing:
        return df
    if final_col not in df.columns:
        df[final_col] = ""
    for c in existing:
        df[c] = df[c].fillna("")
        df[final_col] = df[final_col].fillna("")
        df[final_col] = df[final_col].where(df[final_col] != "", df[c])
    df.drop(columns=[c for c in existing if c != final_col], inplace=True, errors="ignore")
    return df

# loads user provided email dataset from file if it exists
def load_user_provided_data():
    if os.path.exists(USER_PROVIDED_PATH):
        print("[INFO] merging user-provided email dataset...")
        try:
            user_df = pd.read_csv(
                USER_PROVIDED_PATH,
                dtype=str,
                names=["email_text", "email_type", "email_label"]
            )
            print(f"[SUCCESS] loaded {len(user_df)} user-provided emails.")
            return user_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided emails: {e}")
            return pd.DataFrame(columns=["email_text", "email_type", "email_label"])
    return pd.DataFrame(columns=["email_text", "email_type", "email_label"])

# processes unified email dataset and saves cleaned master email dataset
def main():
    print("[INFO] starting master email dataset creation...")
    dataset_path = "unified_email_dataset.csv"
    if not os.path.exists(dataset_path):
        print("[ERROR] unified_email_dataset.csv not found.")
        return
    print("[INFO] processing unified_email_dataset.csv...")
    try:
        df = pd.read_csv(dataset_path, low_memory=True, dtype=str)
        print(f"[SUCCESS] columns found: {df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] could not read unified_email_dataset.csv: {e}")
        return
    df.columns = [col.lower().strip() for col in df.columns]
    df = unify_columns(df, ["email text", "message", "text", "v2", "email_text"], "email_text")
    df = unify_columns(df, ["spam/ham", "sentiment", "label", "v1", "email type", "email_type"], "email_type")
    df.drop(columns=["url", "email", "phone", "message id", "subject", "date", "textid", "selected_text"],
            inplace=True, errors="ignore")
    if "email_text" not in df.columns or "email_type" not in df.columns:
        print(f"[ERROR] missing required columns. found: {df.columns.tolist()}")
        return
    df["email_text"] = df["email_text"].fillna("").astype(str)
    df["email_type"] = df["email_type"].fillna("").astype(str)
    df = df[~df["email_type"].str.lower().eq("unknown")]
    df["email_text"] = df["email_text"].apply(clean_text)
    label_map = {
        "spam": "spam email",
        "smishing": "phishing email",
        "phishing": "phishing email",
        "traditional phishing": "phishing email",
        "spear phishing": "phishing email",
        "ham": "safe email",
        "enron ham": "safe email",
        "hard ham": "safe email",
        "negative": "spam email",
        "positive": "safe email",
        "neutral": "safe email"
    }
    df["email_type"] = df["email_type"].map(lambda x: label_map.get(x.lower().strip(), x))
    df.drop_duplicates(subset=["email_text"], inplace=True)
    df = df[["email_text", "email_type"]]
    df["email_label"] = df["email_type"].map({
        "safe email": 0,
        "spam email": 1,
        "phishing email": 2
    }).fillna(-1).astype(int)
    df.dropna(subset=["email_text", "email_type"], inplace=True)
    invalid_types = {"nan", "please review your account security settings."}
    df = df[~df["email_type"].str.lower().isin(invalid_types)]
    df = df[df["email_label"] != -1]
    user_df = load_user_provided_data()
    if not user_df.empty:
        user_df["email_type"] = user_df["email_type"].fillna("").astype(str)
        user_df = user_df[~user_df["email_type"].str.lower().isin(invalid_types)]
        user_df["email_label"] = pd.to_numeric(user_df["email_label"], errors="coerce").fillna(-1).astype(int)
        user_df = user_df[user_df["email_label"] != -1]
    df = pd.concat([df, user_df], ignore_index=True).drop_duplicates(subset=["email_text"])
    df.to_csv(DATASET_PATH, index=False)
    print(f"[SUCCESS] finished. saved -> {DATASET_PATH}")

if __name__ == "__main__":
    main()
