import streamlit as st
import pandas as pd
import os
from model_loader import predict_email
from url_utils import extract_urls, check_urls

# load the CSV dynamically with caching
@st.cache_data
def load_user_data():
    if os.path.exists("user_provided_emails.csv"):
        return pd.read_csv("user_provided_emails.csv", names=["Email Text", "Label", "Class"], encoding="utf-8")
    return pd.DataFrame(columns=["Email Text", "Label", "Class"])

@st.cache_data
def load_user_urls():
    if os.path.exists("user_provided_urls.csv"):
        return pd.read_csv("user_provided_urls.csv", names=["URL", "Risk Level"], encoding="utf-8")
    return pd.DataFrame(columns=["URL", "Risk Level"])

# store a user-provided email
def store_user_provided_email(email_text, label):
    label_map = {"Safe Email": 0, "Spam Email": 1, "Phishing Email": 2}
    normalized_label = label.title()
    if normalized_label in label_map:
        with open("user_provided_emails.csv", "a", encoding="utf-8") as f:
            f.write(f"{email_text},{normalized_label},{label_map[normalized_label]}\n")
        st.success("User-provided email stored.")
        st.cache_data.clear()
    else:
        st.error("Invalid label. Email not stored.")

# Resets the classification state in the session
def reset_classification():
    st.session_state.predicted = False
    st.session_state.predicted_label = None
    st.session_state.final_prob = None
    st.session_state.warning_message = None

# main Streamlit app
def main():
    st.set_page_config(page_title="Email Classifier", layout="centered")

    user_emails = load_user_data()
    user_urls = load_user_urls()

    st.image("h_logo_white-400x60.png", use_container_width=True)
    st.markdown("<h2 class='gold-text'>Email Classifier</h2>", unsafe_allow_html=True)

    if "predicted" not in st.session_state:
        st.session_state.predicted = False
        st.session_state.user_email = ""

    st.session_state.user_email = st.text_area(
        "Enter an email text to classify:",
        height=200,
        value=st.session_state.user_email,
        key="user_email_input",
        disabled=st.session_state.predicted
    )

    if not st.session_state.predicted:
        if st.button("Classify Email"):
            if not st.session_state.user_email.strip():
                st.error("Please enter some email text first.")
                return
            with st.spinner("Classification in progress..."):
                urls = extract_urls(st.session_state.user_email)
                risk_value, risk_source = check_urls(urls) if urls else (0, "none")
                predicted_label, final_prob, warning_message = predict_email(st.session_state.user_email)

                if risk_value == 2:
                    st.warning("Warning: Unsafe URL detected, automatically flagging the input as unsafe.")
                    predicted_label = "Phishing Email"
                    final_prob = [0.0, 0.0, 1.0]
                    warning_message = f"Detected phishing URL ({risk_source} database). Classification adjusted."
                else:
                    label_map = {"safe email": "Safe Email", "spam email": "Spam Email", "phishing email": "Phishing Email"}
                    predicted_label = label_map.get(predicted_label.lower(), "Unknown")

            st.session_state.update({
                "predicted": True,
                "predicted_label": predicted_label,
                "final_prob": final_prob,
                "warning_message": warning_message,
            })

    if st.session_state.predicted:
        st.markdown("<h2 class='gold-text'>Probability Scores</h2>", unsafe_allow_html=True)
        label_display_map = {0: "Safe Email", 1: "Spam Email", 2: "Phishing Email"}
        for i, score in enumerate(st.session_state["final_prob"]):
            st.write(f"- {label_display_map[i]}: {score:.4f}")

        st.markdown(f"<h3>Final Prediction: <span class='final-prediction'>{st.session_state['predicted_label']}</span></h3>", unsafe_allow_html=True)

        if st.session_state["warning_message"]:
            st.info(st.session_state["warning_message"])

        agree = st.radio("Do you agree with this classification?", ["Yes", "No"], index=0)

        if agree == "No":
            corrected_label = st.radio("Select the correct label:", ["Safe Email", "Spam Email", "Phishing Email"])
            consent = st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state.user_email, corrected_label)
                reset_classification()
        else:
            consent = st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state.user_email, st.session_state["predicted_label"])
                reset_classification()

        if st.button("Try Again"):
            reset_classification()
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # display dynamically updated stored emails
    st.markdown("<h2 class='gold-text'>User-Provided Emails</h2>", unsafe_allow_html=True)
    if not user_emails.empty:
        st.dataframe(user_emails)
        if st.button("Refresh Emails"):
            st.cache_data.clear()
            st.experimental_rerun()

    st.markdown("<h2 class='gold-text'>User-Provided URLs</h2>", unsafe_allow_html=True)
    if not user_urls.empty:
        st.dataframe(user_urls)
        if st.button("Refresh URLs"):
            st.cache_data.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
