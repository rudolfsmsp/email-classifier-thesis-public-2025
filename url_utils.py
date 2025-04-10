import requests
import re
import os
import pandas as pd
import subprocess

MASTER_DATASET_PATH = "master_url_dataset.csv"
CACHE_DIR = "data/external_phishing_checker"
CACHE_FILE = os.path.join(CACHE_DIR, "phishing_urls.txt")
os.makedirs(CACHE_DIR, exist_ok=True)
url_mapping = {}
PHISHING_SOURCES = [
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-INACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-domains-ACTIVE.txt",
]

def normalize_url(url: str) -> str:
    url = url.lower().strip()
    if url.startswith("http://"):
        url = url[len("http://"):]
    elif url.startswith("https://"):
        url = url[len("https://"):]
    return url.rstrip("/")

def load_master_url_dataset(force_reload: bool = False):
    global url_mapping
    if url_mapping and not force_reload:
        print("[INFO] master URL dataset already loaded. Skipping reload.")
        return
    print("[INFO] loading master URL dataset...")
    try:
        url_dataset = pd.read_csv(MASTER_DATASET_PATH, dtype=str, low_memory=False)
        url_dataset.dropna(subset=["url", "label"], inplace=True)
        url_mapping = {
            normalize_url(row["url"]): str(row["label"]).strip()
            for _, row in url_dataset.iterrows()
        }
        print(f"[SUCCESS] loaded {len(url_mapping)} URLs from {MASTER_DATASET_PATH}.")
    except Exception as e:
        print(f"[ERROR] failed to load master dataset ({e}).")
        url_mapping = {}

def extract_urls(text: str) -> list:
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    extracted_domains = set()
    for u in urls:
        try:
            extracted_domains.add(u.split("/")[2])
        except IndexError:
            pass
    return urls + list(extracted_domains)

def load_phishing_urls() -> set:
    print("[INFO] loading cached phishing urls...")
    if not os.path.exists(CACHE_FILE):
        print("[WARNING] phishing_urls.txt not found. external check is empty.")
        return set()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            phishing_urls = set(
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            )
        print(f"[SUCCESS] loaded {len(phishing_urls)} phishing urls from cache.")
        return phishing_urls
    except Exception as e:
        print(f"[ERROR] failed to load phishing database ({e}). returning empty set.")
        return set()

def check_urls(urls: list, email_label=None) -> tuple:
    print("[INFO] checking urls against internal and external databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")
    if not url_mapping:
        load_master_url_dataset()
    phishing_urls = load_phishing_urls()
    for url in urls:
        normalized = normalize_url(url)
        if normalized in url_mapping:
            risk = url_mapping[normalized]
            print(f"[INFO] found url in master dataset: {normalized} | risk: {risk}")
            if risk == "2":
                return (2, "internal")
            else:
                continue
        if normalized in phishing_urls:
            print(f"[INFO] found url in external phishing DB: {normalized}")
            return (2, "external")
    print("[INFO] no threats detected in provided URLs.")
    return (0, "none")

def fetch_phishing_database():
    print("[INFO] fetching external phishing database...")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    all_urls = set()
    for source_url in PHISHING_SOURCES:
        try:
            print(f"[INFO] Downloading from {source_url}")
            response = requests.get(source_url, timeout=30)
            if response.status_code == 200:
                lines = response.text.splitlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        all_urls.add(line.lower().rstrip("/"))
            else:
                print(f"[WARNING] {source_url} returned status code {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Could not fetch {source_url}: {e}")
    if not all_urls:
        print("[WARNING] no phishing URLs downloaded. check your sources or connection.")
        return
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            for url in sorted(all_urls):
                f.write(url + "\n")
        print(f"[SUCCESS] Wrote {len(all_urls)} phishing URLs to {CACHE_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save {CACHE_FILE}: {e}")

if __name__ == "__main__":
    print("[INFO] url utility module ready for use.")
