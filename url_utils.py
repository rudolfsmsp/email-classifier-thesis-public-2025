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

# normalizes a url by removing its protocol and trailing slash
def normalize_url(url):
    url = url.lower().strip()
    if url.startswith("http://"):
        url = url[len("http://"):]
    elif url.startswith("https://"):
        url = url[len("https://"):]
    return url.rstrip("/")

def load_master_url_dataset(force_reload=False):
    global url_mapping
    if url_mapping and not force_reload:
        print("[INFO] master URL dataset already loaded. Skipping reload.")
        return

    print("[INFO] loading master URL dataset...")
    try:
        url_dataset = pd.read_csv(MASTER_DATASET_PATH, dtype=str, low_memory=False)
        url_dataset.dropna(subset=["url", "label"], inplace=True)
        # Create a dictionary { normalized_url: label }
        url_mapping = {
            normalize_url(row["url"]): str(row["label"]).strip()
            for _, row in url_dataset.iterrows()
        }
        print(f"[SUCCESS] loaded {len(url_mapping)} URLs from {MASTER_DATASET_PATH}.")
    except Exception as e:
        print(f"[ERROR] failed to load master dataset ({e}).")
        url_mapping = {}

def extract_urls(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    extracted_domains = set()
    for url in urls:
        try:
            extracted_domains.add(url.split("/")[2])
        except IndexError:
            pass
    return urls + list(extracted_domains)

def load_phishing_urls():
    print("[INFO] loading cached phishing urls...")
    if not os.path.exists(CACHE_FILE):
        # Optionally fetch from external sources if you desire
        return set()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            phishing_urls = set(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))
        print(f"[SUCCESS] loaded {len(phishing_urls)} phishing urls from cache.")
        return phishing_urls
    except Exception as e:
        print(f"[ERROR] failed to load phishing database ({e}). returning empty set.")
        return set()

def check_urls(urls, email_label=None):
    print("[INFO] checking urls against internal and external databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")

    if not url_mapping:
        load_master_url_dataset()

    phishing_urls = load_phishing_urls()

    for url in urls:
        normalized = normalize_url(url)

        # check if in our local master_url_dataset
        if normalized in url_mapping:
            risk = url_mapping[normalized]
            print(f"[INFO] found url in master dataset: {normalized} | risk: {risk}")
            if risk == "2":
                return (2, "internal")
            else:
                continue

        # check if in external phishing list
        if normalized in phishing_urls:
            print(f"[INFO] found url in external phishing DB: {normalized}")
            return (2, "external")

    print("[INFO] no threats detected in provided URLs.")
    return (0, "none")

if __name__ == "__main__":
    print("[INFO] url utility module ready for use.")
