import requests
import re
import os
import pandas as pd
import subprocess
import time

MASTER_DATASET_PATH="master_url_dataset.csv"
USER_PROVIDED_PATH="user_provided_urls.csv"
USER_PROVIDED_EMAILS="user_provided_emails.csv"
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

# loads user-provided emails if available
def load_user_provided_emails():
    if os.path.exists(USER_PROVIDED_EMAILS):
        print("[INFO] loading user-provided email dataset...")
        try:
            email_df=pd.read_csv(USER_PROVIDED_EMAILS,dtype=str,names=["email","label"],on_bad_lines="skip")
            print(f"[SUCCESS] loaded {len(email_df)} user-provided emails.")
            return email_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided emails: {e}")
            return pd.DataFrame(columns=["email","label"])
    return pd.DataFrame(columns=["email","label"])

# writes user-submitted emails safely to user_provided_emails.csv
def save_user_email(email_text,label):
    try:
        with open(USER_PROVIDED_EMAILS,"a") as f:
            f.write(f"{email_text},{label}\n")
        print(f"[SUCCESS] Added user-provided email: {email_text} | Label: {label}")
        subprocess.run(["git","add",USER_PROVIDED_EMAILS])
        subprocess.run(["git","commit","-m",f"Auto-update user emails: {email_text} | Label: {label}"])
        subprocess.run(["git","push","origin","main"])
        print("[SUCCESS] User-provided emails pushed to GitHub.")
    except Exception as e:
        print(f"[ERROR] Failed to save user-provided email: {e}")

    time.sleep(5)  # Ensure emails are updated before processing URLs
    sync_user_urls_with_emails()

# syncs user URLs with the classification from user emails
def sync_user_urls_with_emails():
    email_data = load_user_provided_emails()
    if email_data.empty:
        print("[INFO] No user-provided emails found. Skipping URL sync.")
        return
    latest_email = email_data.iloc[-1]
    latest_label = latest_email["label"]
    urls = extract_urls(latest_email["email"])
    for url in urls:
        if url.strip():
            print(f"[INFO] Syncing URL from email dataset: {url} | Label: {latest_label}")
            save_user_url(url, latest_label)

# writes user-submitted URLs safely to user_provided_urls.csv and auto pushes to GitHub
def save_user_url(url,label):
    url=normalize_url(url)
    existing_data=load_user_provided_data()

    # check if URL already exists
    if url in existing_data["url"].tolist():
        existing_label=existing_data.loc[existing_data["url"]==url,"label"].values[0]
        if existing_label=="2" and label!="2":
            print(f"[WARNING] Attempted to override phishing URL as safe: {url}. Keeping phishing classification.")
            return
        if existing_label!=label:
            existing_data.loc[existing_data["url"]==url,"label"]=label
            existing_data.to_csv(USER_PROVIDED_PATH,index=False)
            print(f"[INFO] Updated classification for {url} to {label}.")
        else:
            print(f"[INFO] User-submitted URL already exists with label: {existing_label}. Skipping.")
        return
    try:
        with open(USER_PROVIDED_PATH,"a") as f:
            f.write(f"{url},{label}\n")
        print(f"[SUCCESS] Added user-submitted URL: {url} | Label: {label}")
        subprocess.run(["git","add",USER_PROVIDED_PATH])
        subprocess.run(["git","commit","-m",f"Auto-update user URLs: {url} | Label: {label}"])
        subprocess.run(["git","push","origin","main"])
        print("[SUCCESS] User-provided URLs pushed to GitHub.")
    except Exception as e:
        print(f"[ERROR] Failed to save user-submitted URL: {e}")

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

# loads phishing urls from the cached file and returns them as a set
def load_phishing_urls():
    print("[INFO] loading cached phishing urls...")
    if not os.path.exists(CACHE_FILE):
        fetch_phishing_database()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            phishing_urls = set(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))
        print(f"[SUCCESS] loaded {len(phishing_urls)} phishing urls from cache.")
        return phishing_urls
    except Exception as e:
        print(f"[ERROR] failed to load phishing database ({e}). returning empty set.")
        return set()

# checks if any extracted url or domain (normalized) is in the master dataset or in the phishing database and returns a risk tuple
def check_urls(urls, email_label=None):
    print("[INFO] checking urls against databases...")
    if not urls:
        print("[INFO] no urls found. skipping check.")
        return (0, "none")
    if not url_mapping:
        load_master_url_dataset()
    phishing_urls=load_phishing_urls()
    user_data=load_user_provided_data()
    user_urls=user_data["url"].tolist()
    for url in urls:
        normalized=normalize_url(url)
        if normalized in user_urls:
            stored_label=user_data.loc[user_data["url"]==normalized,"label"].values[0]
            print(f"[INFO] user-provided URL detected: {normalized}. Using stored label: {stored_label}.")
            return (int(stored_label),"user_submitted")
        if normalized in url_mapping:
            risk=url_mapping[normalized]
            print(f"[INFO] found url in internal database: {normalized} | risk: {risk}")
            return (2,"internal") if risk=="2" else (0,"internal")
        print(f"[INFO] New URL detected. Assigning label from email: {email_label}")
        save_user_url(normalized, email_label if email_label else "0")  # Assign same label as email
    print("[INFO] no threats detected in provided URLs.")
    return (0,"none")

if __name__=="__main__":
    print("[INFO] url utility module ready for use.")
