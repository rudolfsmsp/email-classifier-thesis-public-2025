import streamlit as st
import pandas as pd
import os
import subprocess
import time
import csv
from model_loader import predict_email
from url_utils import extract_urls, check_urls

import pandas as pd
import os
import subprocess
import csv

def store_user_provided_email(email_text, label):
    label_map = {"Safe Email": 0, "Spam Email": 1, "Phishing Email": 2}
    normalized_label = label.title()
    file_path = "user_provided_emails.csv"

    if normalized_label in label_map:
        try:
            existing_entries = set()
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    existing_entries = {tuple(row) for row in reader if len(row) == 3}  # Only valid rows
            clean_email_text = email_text.replace(",", " ")
            new_entry = (clean_email_text, normalized_label, str(label_map[normalized_label]))
            if new_entry in existing_entries:
                st.warning("This email has already been stored.")
                return
            with open(file_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_entry)
            st.success("User-provided email stored.")
            subprocess.run(["git", "add", "user_provided_emails.csv"])
            result = subprocess.run(["git", "diff", "--cached", "--quiet"])  # Check if changes exist
            if result.returncode != 0:  # If changes exist
                subprocess.run(["git", "commit", "-m", "Auto-update user emails"], check=True)
                subprocess.run(["git", "push", "origin", "main"], check=True)
                st.success("Changes pushed to GitHub.")
            else:
                st.info("No new changes to push.")
        except Exception as e:
            st.error(f"Error processing email storage: {e}")
    else:
        st.error("Invalid label. Email not stored.")


def store_user_provided_urls(urls, risk_value, risk_source):
            if not urls:
                return
            file_path = "user_provided_urls.csv"
            existing_urls = set()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_urls = {line.strip().split(",")[0] for line in f if line.strip()}
            except FileNotFoundError:
                pass
            new_entries = []
            with open(file_path, "a", encoding="utf-8") as f:
                for url in urls:
                    if url not in existing_urls:
                        entry = f"{url},{risk_value},{risk_source}\n"
                        f.write(entry)
                        new_entries.append(entry)
            if new_entries:
                st.success("User-provided URLs stored.")
                subprocess.run(["git", "add", "user_provided_urls.csv"])
                result = subprocess.run(["git", "diff", "--cached", "--quiet"])  # Check if there are changes
                if result.returncode != 0:  # If changes exist
                    subprocess.run(["git", "commit", "-m", "Auto-update user-provided URLs"], check=True)
                    subprocess.run(["git", "push", "origin", "main"], check=True)
                    st.success("URL changes pushed to GitHub.")
                else:
                    st.info("No new URL changes to push.")

def reset_classification():
    st.session_state.predicted = False
    st.session_state.predicted_label = None
    st.session_state.final_prob = None
    st.session_state.warning_message = None
    st.session_state.retraining = False

def retrain_model():
    st.session_state.retraining = True  # Start loading
    st.info("Starting model retraining...")

    with st.spinner("Retraining model..."):
        try:
            start_time = time.time()
            subprocess.run(["python3", "create_master_email_dataset.py"], check=True)
            subprocess.run(["python3", "create_master_url_dataset.py"], check=True)
            subprocess.run(["python3", "train_email_classifier.py"], check=True)
            end_time = time.time()
            st.success(f"Model retrained successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            st.error(f"Retraining failed: {e}")

    st.session_state.retraining = False

def main():
    st.set_page_config(page_title="Email Classifier",layout="centered")
    st.markdown(
        """
        <style>
        body, html, .stApp {font-family: 'Montserrat', sans-serif;background-color: #2B2B2B;}
        textarea {background-color: #000 !important;color: #fff !important;}
        textarea::placeholder {color: #999999 !important;}
        h1, h2, h3, h4, h5, h6 {color: rgb(218,165,33) !important;font-weight: 700;}
        p, label, span, div, button, .stMarkdown, .stRadio, .stCheckbox {color: #FFFFFF;}
        .main-container {background-color: #2B2B2B;border-radius: 10px;padding: 2rem;box-shadow: none;margin-top: 0.5rem;}
        .stButton > button {background-color: rgb(218,165,33) !important;color: #ffffff !important;border-radius: 6px;padding: 0.6em 1em;font-weight: 600;}
        .final-prediction {color: #00FF00 !important;font-weight: bold;}
        .gold-text {color: rgb(218,165,33) !important;font-weight: 600;}
        .notice-box {border: 1px solid rgb(218,165,33);padding: 1rem;margin-top: 1rem;margin-bottom: 1rem;font-style: italic;color: #FFFFFF;}
        </style>
        """,unsafe_allow_html=True)
    st.image("h_logo_white-400x60.png",use_container_width=True)
    st.markdown('<div class="main-container">',unsafe_allow_html=True)
    st.write("Hi, my name is **Rūdolfs Mušperts**, and this is my Email Classifier Project for Bachelor Thesis.")
    st.markdown(
        """
        This web app classifies email messages into three categories: 
        <span class='gold-text'>safe email</span>, <span class='gold-text'>spam email</span>, or <span class='gold-text'>phishing email</span>.
        It also checks for any URLs in the message and flags potential phishing links. When you type an email text below and click 
        **Classify Email**, the system uses trained machine learning models to predict the category.
        If you disagree with the prediction, you can override it. If you consent, your email text and final label will be stored in a CSV file (**user_provided_emails.csv**) 
        to improve the model. Any suspicious URLs you provide will be stored in **user_provided_urls.csv**.
        """,unsafe_allow_html=True)
    st.markdown(
        """
        <span class='gold-text'>Safe email</span> – is a genuine message from a trusted sender that doesn't try to trick you or steal your information.<br><br>
        <span class='gold-text'>Spam email</span> – is an unwanted message filled with ads or promotions that clutters your inbox, but usually doesn’t ask for personal details.<br><br>
        <span class='gold-text'>Phishing email</span> – is a scam that pretends to be from a reputable company or bank and uses urgent language to try to get you to reveal sensitive information like passwords or credit card numbers.<br><br>
        **How to differentiate between spam and phishing**:<br>
        - Spam emails are mostly commercial and sent in bulk, without trying to steal personal data.<br>
        - Phishing emails use urgent warnings, fake links, or requests for sensitive information to deceive you.
        """,unsafe_allow_html=True)
    st.markdown(
        """
        <div class="notice-box">
        Please separate links from hyperlinked text, as the tool does not work with links hidden behind hyperlink text.
        </div>
        """,unsafe_allow_html=True)
    st.markdown("<h2 class='gold-text'>Email Classifier</h2>",unsafe_allow_html=True)
    if "predicted" not in st.session_state:
        st.session_state.predicted=False
        st.session_state.user_email=""
    st.session_state.user_email=st.text_area("Enter an email text to classify:",height=200,value=st.session_state.user_email,key="user_email_input",disabled=st.session_state.predicted)
    if not st.session_state.predicted:
        if st.button("Classify Email"):
            if not st.session_state.user_email.strip():
                st.error("Please enter some email text first.")
                return
            with st.spinner("Classification in-progress..."):
                urls=extract_urls(st.session_state.user_email)
                risk_value, risk_source=check_urls(urls) if urls else (0,"none")
                predicted_label,final_prob,warning_message=predict_email(st.session_state.user_email)
                if risk_value==2:
                    st.warning("Warning: Unsafe URL detected, automatically flagging the input as unsafe.")
                    predicted_label="Phishing Email"
                    final_prob=[0.0,0.0,1.0]
                    warning_message=f"detected phishing url ({risk_source} database). classification adjusted."
                else:
                    label_map={"safe email":"Safe Email","spam email":"Spam Email","phishing email":"Phishing Email"}
                    predicted_label=label_map.get(predicted_label.lower(),"Unknown")
            st.session_state.update({"predicted":True,"predicted_label":predicted_label,"final_prob":final_prob,"warning_message":warning_message,"urls":urls,"risk_value":risk_value,"risk_source":risk_source})
    if st.session_state.predicted:
        st.markdown("<h2 class='gold-text'>Probability Scores</h2>",unsafe_allow_html=True)
        label_display_map={0:"Safe Email",1:"Spam Email",2:"Phishing Email"}
        for i,score in enumerate(st.session_state["final_prob"]):
            st.write(f"- {label_display_map[i]}: {score:.4f}")
        st.markdown(f"<h3>Final Prediction: <span class='final-prediction'>{st.session_state['predicted_label']}</span></h3>",unsafe_allow_html=True)
        if st.session_state["warning_message"]:
            st.info(st.session_state["warning_message"])
        agree=st.radio("Do you agree with this classification?",["Yes","No"],index=0)
        if agree=="No":
            corrected_label=st.radio("Select the correct label:",["Safe Email","Spam Email","Phishing Email"])
            consent=st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state.user_email,corrected_label)
                if st.session_state["risk_value"]==2:
                    store_user_provided_urls(st.session_state["urls"],st.session_state["risk_value"],st.session_state["risk_source"])
                reset_classification()
        else:
            consent=st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state.user_email,st.session_state["predicted_label"])
                if st.session_state["risk_value"]==2:
                    store_user_provided_urls(st.session_state["urls"],st.session_state["risk_value"],st.session_state["risk_source"])
                reset_classification()
        if not st.session_state.get("retraining", False):
            if st.button("Try Again"):
                reset_classification()
                st.rerun()

    st.markdown("</div>",unsafe_allow_html=True)

if __name__=="__main__":
    main()
