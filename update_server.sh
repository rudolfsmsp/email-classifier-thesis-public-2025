#!/bin/bash

cd ~/email-classifier-thesis-public-2025 || exit

echo "[INFO] Pulling latest changes from GitHub..."
git fetch --all
git reset --hard origin/main
git pull origin main

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Check if new user data exists
if git diff --quiet user_provided_emails.csv ; then
    echo "[INFO] No new user data to push."
else
    echo "[INFO] New user data found. Committing changes..."
    git add user_provided_emails.csv
    git commit -m "Auto-update: new user-provided emails and URLs"
    git push origin main
    echo "[SUCCESS] User data pushed to GitHub."
fi

# Retrain the model every 30 minutes
echo "[INFO] Retraining the model..."
python3 create_master_email_dataset.py
python3 create_master_url_dataset.py
python3 train_email_classifier.py
echo "[SUCCESS] Model retrained successfully."

# Restart Streamlit
pkill -f streamlit
nohup .venv/bin/streamlit run interface.py --server.port 8502 --server.address 0.0.0.0 --server.enableCORS false > streamlit.log 2>&1 &

echo "[SUCCESS] Server updated, retrained, and restarted successfully."

