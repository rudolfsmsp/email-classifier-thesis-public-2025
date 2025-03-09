import mailbox
import pandas as pd
import os
import subprocess
import platform

data_dir = "data/diegoocampoh_emails"
mbox_files = {
    "enron": os.path.join(data_dir, "emails-enron.mbox"),
    "phishing": os.path.join(data_dir, "emails-phishing.mbox")
}
csv_files = {
    "enron": os.path.join(data_dir, "emails-enron.csv"),
    "phishing": os.path.join(data_dir, "emails-phishing.csv")
}
mbox_urls = {
    "enron": "https://raw.githubusercontent.com/diegoocampoh/MachineLearningPhishing/master/code/resources/emails-enron.mbox",
    "phishing": "https://raw.githubusercontent.com/diegoocampoh/MachineLearningPhishing/master/code/resources/emails-phishing.mbox"
}

# downloads mbox files from urls
def download_mbox_files():
    print("[INFO] starting download of mbox files...")
    os.makedirs(data_dir, exist_ok=True)
    for dataset, url in mbox_urls.items():
        file_path = mbox_files[dataset]
        try:
            if platform.system() == "Windows":
                subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri {url} -OutFile {file_path}"], check=True)
            else:
                subprocess.run(["wget", "-O", file_path, url], check=True)
            print(f"[SUCCESS] finished downloading {dataset} mbox file.")
        except Exception as e:
            print(f"[ERROR] failed downloading {dataset} mbox file: {e}")

# converts mbox files to csv format
def convert_mbox_to_csv(mbox_files, csv_files):
    print("[INFO] starting conversion of mbox files to csv...")
    for dataset, mbox_path in mbox_files.items():
        if not os.path.exists(mbox_path):
            print(f"[ERROR] file not found: {mbox_path}")
            continue
        try:
            mbox = mailbox.mbox(mbox_path)
            emails = []
            for message in mbox:
                subject = message["subject"] if message["subject"] else "No Subject"
                body = message.get_payload(decode=True)
                body = body.decode(errors="ignore") if body else "No Content"
                emails.append({"email_subject": subject, "email_body": body})
            df = pd.DataFrame(emails)
            df.to_csv(csv_files[dataset], index=False)
            print(f"[SUCCESS] finished processing {mbox_path}. Saved -> {csv_files[dataset]}")
        except Exception as e:
            print(f"[ERROR] failed processing {mbox_path}: {e}")

# processes mbox to csv conversion by downloading and converting files
def main():
    print("[INFO] starting mbox to csv conversion...")
    download_mbox_files()
    convert_mbox_to_csv(mbox_files, csv_files)
    print("[SUCCESS] finished mbox to csv conversion.")

if __name__ == "__main__":
    main()
