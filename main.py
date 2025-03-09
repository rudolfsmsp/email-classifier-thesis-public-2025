import subprocess
import time

# processes environment setup and launches the interface
def main():
    print("[INFO] starting environment setup...")
    subprocess.run(["python", "create_unified_email_dataset.py"])
    subprocess.run(["python", "create_master_email_dataset.py"])
    subprocess.run(["python", "create_unified_url_dataset.py"])
    subprocess.run(["python", "create_master_url_dataset.py"])
    subprocess.run(["python", "train_email_classifier.py"])
    print("[INFO] environment setup complete. launching interface...")
    subprocess.Popen(["streamlit", "run", "interface.py", "--server.headless", "true", "--server.port", "8502"])
    print("[SUCCESS] interface launched successfully.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] exiting...")

if __name__ == "__main__":
    main()