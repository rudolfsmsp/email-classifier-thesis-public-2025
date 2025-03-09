import joblib
import numpy as np
from url_utils import extract_urls, check_urls

# predicts the classification of an email based on text content and url risk
def predict_email(email_text):
    print("[INFO] starting email classification...")
    try:
        nb_tfidf = joblib.load("naive_bayes_tfidf_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        nb_bow = joblib.load("naive_bayes_bow_model.pkl")
        bow_vectorizer = joblib.load("bow_vectorizer.pkl")
    except Exception as e:
        print(f"[ERROR] failed to load model files: {e}")
        return None, None, "failed to load models."
    urls = extract_urls(email_text)
    url_risk = check_urls(urls) if urls else 0
    email_tfidf = tfidf_vectorizer.transform([email_text])
    email_bow = bow_vectorizer.transform([email_text])
    prob_tfidf = nb_tfidf.predict_proba(email_tfidf)[0]
    prob_bow = nb_bow.predict_proba(email_bow)[0]
    if len(prob_tfidf) == 2:
        prob_tfidf = np.append(prob_tfidf, [0])
    if len(prob_bow) == 2:
        prob_bow = np.append(prob_bow, [0])
    final_prob = (prob_tfidf + prob_bow) / 2
    label_mapping = {0: "safe email", 1: "spam email", 2: "phishing email"}
    predicted_label = label_mapping[np.argmax(final_prob)]
    warning_message = ""
    if url_risk == 2:
        predicted_label = "phishing email"
        final_prob[2] = 1.0
        warning_message = "detected phishing url from database. classification adjusted."
    print(f"[SUCCESS] finished email classification. predicted: {predicted_label}")
    return predicted_label, final_prob, warning_message
