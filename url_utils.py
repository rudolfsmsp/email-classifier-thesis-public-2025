import requests
import re
import os
import pandas as pd

MASTER_DATASET_PATH = "master_url_dataset.csv"
USER_PROVIDED_PATH = "user_provided_urls.csv"
PHISHING_URLS = [
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-INACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-domains-ACTIVE.txt"
]
CACHE_DIR = "data/external_phishing_checker"
CACHE_FILE = os.path.join(CACHE_DIR, "phishing_urls.txt")
os.makedirs(CACHE_DIR, exist_ok=True)
url_mapping = {}

# loads the master url dataset and updates the global url_mapping dictionary
def load_master_url_dataset():
    global url_mapping
    print("[INFO] starting master url dataset creation...")
    try:
        url_dataset = pd.read_csv(MASTER_DATASET_PATH, dtype=str, low_memory=False)
        user_dataset = pd.read_csv(USER_PROVIDED_PATH, dtype=str, low_memory=False) if os.path.exists(USER_PROVIDED_PATH) else pd.DataFrame(columns=["url", "label"])
        combined_dataset = pd.concat([url_dataset, user_dataset], ignore_index=True).drop_duplicates(subset=["url"])
        url_mapping = {row["url"]: int(float(row["label"])) for _, row in combined_dataset.iterrows()}
        print(f"[SUCCESS] loaded {len(url_mapping)} urls from combined master dataset.")
    except Exception as e:
        print(f"[ERROR] failed to load master dataset ({e}).")
        url_mapping = {}

# downloads the latest phishing database files and saves them to the cache file
def fetch_phishing_database():
    print("[INFO] fetching latest phishing database...")
    phishing_data = set()
    for source_url in PHISHING_URLS:
        try:
            print(f"[INFO] fetching data from {source_url}...")
            response = requests.get(source_url, timeout=10)
            response.raise_for_status()
            phishing_data.update(line.strip() for line in response.text.splitlines())
        except Exception as e:
            print(f"[ERROR] failed to fetch phishing database from {source_url} ({e}). skipping this source.")
    if phishing_data:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(phishing_data))
        print(f"[SUCCESS] phishing database updated. saved at {CACHE_FILE}")
    else:
        print("[ERROR] all phishing database sources failed.")

# loads phishing urls from the cached file and returns them as a set
def load_phishing_urls():
    print("[INFO] loading cached phishing urls...")
    if not os.path.exists(CACHE_FILE):
        fetch_phishing_database()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            phishing_urls = set(line.strip() for line in f)
        print(f"[SUCCESS] loaded {len(phishing_urls)} phishing urls from cache.")
        return phishing_urls
    except Exception as e:
        print(f"[ERROR] failed to load phishing database ({e}). returning empty set.")
        return set()

# extracts urls and domains from the given text and returns a list of both
def extract_urls(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    extracted_domains = set()
    for url in urls:
        try:
            extracted_domains.add(url.split("/")[2])
        except IndexError:
            pass
    return urls + list(extracted_domains)

# checks if any extracted url or domain is in the master dataset or in the phishing database and returns a risk tuple
def check_urls(urls):
    print("[INFO] checking urls against databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")
    if not url_mapping:
        load_master_url_dataset()
    phishing_urls = load_phishing_urls()
    for url in urls:
        domain = url.split("/")[2] if "://" in url else url
        if url in url_mapping:
            risk = url_mapping[url]
            print(f"[INFO] found url in internal database: {url} | risk: {risk}")
            if risk == 2:
                return (2, "internal")
            else:
                return (0, "internal")
        if domain in phishing_urls:
            print(f"[INFO] detected phishing domain from external database: {domain}")
            with open(USER_PROVIDED_PATH, "a") as f:
                f.write(f"{url},2\n")
            print(f"[INFO] added {url} to internal phishing database.")
            return (2, "external")
    print("[INFO] no threats detected in provided urls.")
    return (0, "none")

if __name__ == "__main__":
    print("[INFO] url utility module ready for use.")
