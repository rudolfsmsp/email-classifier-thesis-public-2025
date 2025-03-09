import streamlit as st
from model_loader import predict_email
from url_utils import extract_urls, check_urls

# stores a user-provided email with its label into a csv file
def store_user_provided_email(email_text, label):
    label_map = {"Safe Email": 0, "Spam Email": 1, "Phishing Email": 2}
    normalized_label = label.title()
    if normalized_label in label_map:
        with open("user_provided_emails.csv", "a", encoding="utf-8") as f:
            f.write(f"{email_text},{normalized_label},{label_map[normalized_label]}\n")
        st.success("User-provided email stored.")
    else:
        st.error("Invalid label. Email not stored.")

# resets the classification state in the session
def reset_classification():
    st.session_state.predicted = False
    st.session_state.predicted_label = None
    st.session_state.final_prob = None
    st.session_state.warning_message = None

# runs the streamlit app for email classification
def main():
    st.set_page_config(page_title="Email Classifier", layout="centered")
    st.markdown(
        """
        <style>
        #MainMenu, header, footer { visibility: hidden; }
        body, html, .stApp {
            font-family: 'Montserrat', sans-serif;
            background-color: #1B1B1B;
        }
        textarea { background-color: #000 !important; color: #fff !important; }
        h1, h2, h3, h4, h5, h6 {
            color: rgb(218,165,33);
            font-weight: 700;
        }
        p, label, span, div, button, .stMarkdown, .stRadio, .stCheckbox {
            color: #FFFFFF;
        }
        .main-container {
            background-color: #2B2B2B;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
        }
        .stButton > button {
            background-color: rgb(218,165,33);
            color: #ffffff;
            border-radius: 6px;
            padding: 0.6em 1em;
            font-weight: 600;
        }
        .final-prediction { color: #00FF00; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("h_logo_white-400x60.png", use_container_width=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.write("Hi, my name is **Rūdolfs Mušperts**, and this is my Email Classifier Project for Bachelor Thesis.")
    st.write(
        """
        This web app classifies email messages into three categories:
        **safe email**, **spam email**, or **phishing email**.
        It also checks for any URLs in the message and flags potential phishing links.
        When you type an email text below and click **Classify Email**,
        the system uses trained machine learning models to predict the category.
        If you disagree with the prediction, you can override it.
        If you consent, your email text and final label will be stored
        in a CSV file (**user_provided_emails.csv**) to improve the model.
        """,
        unsafe_allow_html=True
    )
    st.write("## Email Classifier")
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
            urls = extract_urls(st.session_state.user_email)
            risk_value, risk_source = check_urls(urls) if urls else (0, "none")
            predicted_label, final_prob, warning_message = predict_email(st.session_state.user_email)
            if risk_value == 2:
                predicted_label = "Phishing Email"
                final_prob = [0.0, 0.0, 1.0]
                warning_message = f"detected phishing url ({risk_source} database). classification adjusted."
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
        label_display_map = {0: "Safe Email", 1: "Spam Email", 2: "Phishing Email"}
        st.write("### Probability Scores:")
        for i, score in enumerate(st.session_state["final_prob"]):
            st.write(f"- {label_display_map[i]}: {score:.4f}")
        st.markdown(
            f"### Final Prediction: <span class='final-prediction'>{st.session_state['predicted_label']}</span>",
            unsafe_allow_html=True,
        )
        if st.session_state["warning_message"]:
            st.info(st.session_state["warning_message"])
        agree = st.radio("Do you agree with this classification?", ["Yes", "No"], index=0)
        if agree == "No":
            corrected_label = st.radio("Select the correct label:", ["Safe Email", "Spam Email", "Phishing Email"])
            consent = st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state["user_email"], corrected_label)
                reset_classification()
        else:
            consent = st.checkbox("I consent to adding this email to the dataset for learning.")
            if consent:
                store_user_provided_email(st.session_state["user_email"], st.session_state["predicted_label"])
                reset_classification()
        if st.button("Try Again"):
            reset_classification()
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
