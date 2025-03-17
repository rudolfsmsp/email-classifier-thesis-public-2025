import requests
import re
import os
import pandas as pd
import subprocess

MASTER_DATASET_PATH="master_url_dataset.csv"
USER_PROVIDED_PATH="user_provided_urls.csv"
PHISHING_URLS=[
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-INACTIVE.txt",
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-domains-ACTIVE.txt",
    "https://raw.githubusercontent.com/Zaczero/pihole-phishtank/main/hosts.txt"
]
CACHE_DIR="data/external_phishing_checker"
CACHE_FILE=os.path.join(CACHE_DIR,"phishing_urls.txt")
os.makedirs(CACHE_DIR,exist_ok=True)
url_mapping={}

# normalizes a url by removing its protocol and trailing slash
def normalize_url(url):
    url=url.lower().strip()
    if url.startswith("http://"):
        url=url[len("http://"):]
    elif url.startswith("https://"):
        url=url[len("https://"):]
    return url.rstrip("/")

# loads the master url dataset and updates the global url_mapping dictionary with normalized keys
def load_master_url_dataset(force_reload=False):
    global url_mapping
    if url_mapping and not force_reload:
        print("[INFO] master URL dataset already loaded. Skipping reload.")
        return
    print("[INFO] loading master URL dataset...")
    try:
        url_dataset=pd.read_csv(MASTER_DATASET_PATH,dtype=str,low_memory=False)
        user_dataset=load_user_provided_data()
        combined_dataset=pd.concat([url_dataset,user_dataset],ignore_index=True).drop_duplicates(subset=["url"])
        url_mapping={normalize_url(row["url"]):str(row["label"]).strip() for _,row in combined_dataset.iterrows() if pd.notna(row["url"])}
        print(f"[SUCCESS] loaded {len(url_mapping)} URLs from combined master dataset.")
    except Exception as e:
        print(f"[ERROR] failed to load master dataset ({e}).")
        url_mapping={}

# loads user-provided URLs if available
def load_user_provided_data():
    if os.path.exists(USER_PROVIDED_PATH):
        print("[INFO] loading user-provided URL dataset...")
        try:
            user_df=pd.read_csv(USER_PROVIDED_PATH,dtype=str,names=["url","label"],on_bad_lines="skip")
            user_df["url"]=user_df["url"].fillna("").astype(str).str.lower()
            user_df["label"]=user_df["label"].fillna("").astype(str)
            print(f"[SUCCESS] loaded {len(user_df)} user-provided URLs.")
            return user_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided URLs: {e}")
            return pd.DataFrame(columns=["url","label"])
    return pd.DataFrame(columns=["url","label"])

# writes user-submitted URLs safely to user_provided_urls.csv and auto pushes to GitHub
def save_user_url(url,label):
    url=normalize_url(url)
    existing_data=load_user_provided_data()
    if url in existing_data["url"].tolist():
        print(f"[INFO] user-submitted URL already exists: {url}. Skipping.")
        return
    try:
        with open(USER_PROVIDED_PATH,"a") as f:
            f.write(f"{url},{label}\n")
        print(f"[SUCCESS] added user-submitted URL: {url} | label: {label}")
        subprocess.run(["git","add",USER_PROVIDED_PATH])
        subprocess.run(["git","commit","-m",f"Auto-update user URLs: {url}"])
        subprocess.run(["git","push","origin","main"])
        print("[SUCCESS] user-provided URLs pushed to GitHub.")
    except Exception as e:
        print(f"[ERROR] failed to save user-submitted URL: {e}")

# downloads the latest phishing database files and saves them to the cache file
def fetch_phishing_database():
    print("[INFO] fetching latest phishing database...")
    phishing_data=set()
    for source_url in PHISHING_URLS:
        try:
            print(f"[INFO] fetching data from {source_url}...")
            response=requests.get(source_url,timeout=10)
            response.raise_for_status()
            phishing_data.update(line.strip() for line in response.text.splitlines() if line.strip() and not line.strip().startswith("#"))
        except Exception as e:
            print(f"[ERROR] failed to fetch phishing database from {source_url} ({e}). skipping this source.")
    if phishing_data:
        with open(CACHE_FILE,"w",encoding="utf-8") as f:
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
        with open(CACHE_FILE,"r",encoding="utf-8") as f:
            phishing_urls=set(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))
        print(f"[SUCCESS] loaded {len(phishing_urls)} phishing urls from cache.")
        return phishing_urls
    except Exception as e:
        print(f"[ERROR] failed to load phishing database ({e}). returning empty set.")
        return set()

# extracts urls and domains from the given text and returns a list of both
def extract_urls(text):
    urls=re.findall(r"https?://\S+|www\.\S+",text)
    extracted_domains=set()
    for url in urls:
        try:
            extracted_domains.add(url.split("/")[2])
        except IndexError:
            pass
    return urls+list(extracted_domains)

# checks if any extracted url or domain (normalized) is in the master dataset or in the phishing database and returns a risk tuple
def check_urls(urls):
    print("[INFO] checking urls against databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")
    if not url_mapping:
        load_master_url_dataset()
    phishing_urls = load_phishing_urls()
    user_urls = load_user_provided_data()["url"].tolist()
    for url in urls:
        normalized = normalize_url(url)
        if normalized in user_urls:
            stored_label = url_mapping.get(normalized, "0")
            print(f"[INFO] user-provided URL detected: {normalized}. Using stored label: {stored_label}")
            return (int(stored_label), "user_submitted")
        if normalized in url_mapping:
            risk = url_mapping[normalized]
            print(f"[INFO] found URL in internal database: {normalized} | risk: {risk}")
            return (2, "internal") if risk == "2" else (0, "internal")
        try:
            domain = url.split("/")[2]
            domain = normalize_url(domain)
        except Exception:
            domain = normalized
        if domain in phishing_urls:
            print(f"[INFO] detected phishing domain from external database: {domain}")
            save_user_url(domain, "2") 
            return (2, "external")
        print(f"[INFO] new safe URL detected and stored: {normalized}")
        save_user_url(normalized, "0")
    print("[INFO] no threats detected in provided URLs.")
    return (0, "none")

if __name__=="__main__":
    print("[INFO] url utility module ready for use.")
