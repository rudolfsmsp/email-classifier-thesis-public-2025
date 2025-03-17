import requests
import re
import os
import pandas as pd

MASTER_DATASET_PATH = "master_url_dataset.csv"
USER_PROVIDED_PATH = "user_provided_urls.csv"
PHISHING_URLS = [
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-INACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-domains-ACTIVE.txt",
    "https://raw.githubusercontent.com/Zaczero/pihole-phishtank/main/hosts.txt"
]
CACHE_DIR = "data/external_phishing_checker"
CACHE_FILE = os.path.join(CACHE_DIR, "phishing_urls.txt")
os.makedirs(CACHE_DIR, exist_ok=True)
url_mapping = {}

# normalizes a url by removing its protocol and trailing slash
def normalize_url(url):
    url = url.lower().strip()
    if url.startswith("http://"):
        url = url[len("http://"):]
    elif url.startswith("https://"):
        url = url[len("https://"):]
    return url.rstrip("/")

# loads the master url dataset and updates the global url_mapping dictionary with normalized keys
def load_master_url_dataset(force_reload=False):
    global url_mapping
    if url_mapping and not force_reload:
        print("[INFO] master URL dataset already loaded. Skipping reload.")
        return
    print("[INFO] loading master URL dataset...")
    try:
        url_dataset = pd.read_csv(MASTER_DATASET_PATH, dtype=str, low_memory=False)
        user_dataset = load_user_provided_data()
        combined_dataset = pd.concat([url_dataset, user_dataset], ignore_index=True).drop_duplicates(subset=["url"])
        url_mapping = {normalize_url(row["url"]): str(row["label"]).strip() for _, row in combined_dataset.iterrows() if pd.notna(row["url"])}
        print(f"[SUCCESS] loaded {len(url_mapping)} URLs from combined master dataset.")
    except Exception as e:
        print(f"[ERROR] failed to load master dataset ({e}).")
        url_mapping = {}

# loads user-provided URLs if available
def load_user_provided_data():
    if os.path.exists(USER_PROVIDED_PATH):
        print("[INFO] loading user-provided URL dataset...")
        try:
            user_df = pd.read_csv(USER_PROVIDED_PATH, dtype=str, names=["url", "label"])
            user_df["url"] = user_df["url"].fillna("").astype(str).str.lower()
            user_df["label"] = user_df["label"].fillna("").astype(str)
            print(f"[SUCCESS] loaded {len(user_df)} user-provided URLs.")
            return user_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided URLs: {e}")
            return pd.DataFrame(columns=["url", "label"])
    return pd.DataFrame(columns=["url", "label"])

# writes user-submitted URLs safely to user_provided_urls.csv
def save_user_url(url, label):
    url = normalize_url(url)
    existing_data = load_user_provided_data()

    # Check if the URL is already stored
    if url in existing_data["url"].tolist():
        print(f"[INFO] user-submitted URL already exists: {url}. Skipping.")
        return

    try:
        with open(USER_PROVIDED_PATH, "a") as f:
            f.write(f"{url},{label}\n")
        print(f"[SUCCESS] added user-submitted URL: {url} | label: {label}")
    except Exception as e:
        print(f"[ERROR] failed to save user-submitted URL: {e}")

# checks if any extracted url or domain (normalized) is in the master dataset or in the phishing database and returns a risk tuple
def check_urls(urls):
    print("[INFO] checking urls against databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")
    if not url_mapping:
        load_master_url_dataset()
    phishing_urls = load_phishing_urls()
    user_urls = load_user_provided_data()["url"].tolist()  # Load user-submitted URLs
    for url in urls:
        normalized = normalize_url(url)
        # User-submitted URLs are always stored and marked safe
        if normalized in user_urls:
            print(f"[INFO] user-provided URL detected: {normalized}. Marking as safe.")
            return (0, "user_submitted")
        if normalized in url_mapping:
            risk = url_mapping[normalized]
            print(f"[INFO] found url in internal database: {normalized} | risk: {risk}")
            return (2, "internal") if risk == "2" else (0, "internal")
        try:
            domain = url.split("/")[2]
            domain = normalize_url(domain)
        except Exception:
            domain = normalized
        if domain in phishing_urls:
            if domain in user_urls:
                print(f"[INFO] user-submitted URL detected in phishing database: {domain}. Overriding classification to safe.")
                return (0, "user_submitted")
            print(f"[INFO] detected phishing domain from external database: {domain}")
            save_user_url(domain, "2")  # Store phishing URL properly
            return (2, "external")
        save_user_url(normalized, "0")  # Store safe URL properly
        print(f"[INFO] new safe URL detected and stored: {normalized}")
    print("[INFO] no threats detected in provided URLs.")
    return (0, "none")

if __name__ == "__main__":
    print("[INFO] url utility module ready for use.")
