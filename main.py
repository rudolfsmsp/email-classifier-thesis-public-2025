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
            print(f"[INFO] Running {script}...")
            result = subprocess.run(["python", script], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[ERROR] {script} failed with error:\n{result.stderr}")
            else:
                print(f"[SUCCESS] {script} completed.")
        else:
            print(f"[WARNING] {script} not found, skipping.")

def launch_streamlit():
    print("[INFO] Launching Streamlit app...")
    subprocess.run(["streamlit", "run", "interface.py", "--server.headless", "true"])

if __name__ == "__main__":
    setup_environment()
    launch_streamlit()
