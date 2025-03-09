import subprocess
import os

def setup_environment():
    print("[INFO] Running dataset preparation...")
    scripts = [
        "create_unified_email_dataset.py",
        "create_master_email_dataset.py",
        "create_unified_url_dataset.py",
        "create_master_url_dataset.py",
        "train_email_classifier.py"
    ]

    for script in scripts:
        if os.path.exists(script):
            print(f"[INFO] running {script} in the background...")
            subprocess.Popen(["python", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"[WARNING] {script} not found, skipping.")


def launch_streamlit():
    print("[INFO] Launching Streamlit app...")
    subprocess.run(["streamlit", "run", "interface.py", "--server.headless", "true", "--server.port", "8502"])

if __name__ == "__main__":
    setup_environment()  # Run preprocessing in the background
    launch_streamlit()  # Start Streamlit as the main process
