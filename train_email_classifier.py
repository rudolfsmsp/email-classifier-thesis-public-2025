import pandas as pd
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
from url_utils import extract_urls, check_urls

DATASET_PATH = "master_email_dataset.csv"
USER_PROVIDED_PATH = "master_provided_emails.csv"
MODEL_TFIDF = "naive_bayes_tfidf_model.pkl"
MODEL_BOW = "naive_bayes_bow_model.pkl"
VECTORIZER_TFIDF = "tfidf_vectorizer.pkl"
VECTORIZER_BOW = "bow_vectorizer.pkl"

SPAM_KEYWORDS = set([
    "ecommerce", "buy", "buy direct", "buy today", "clearance", "as seen on",
    "order", "order status", "sample", "wants credit card", "claim now", "act now",
    "this won’t last", "expires soon", "limited time", "exclusive deal", "urgent",
    "100%", "50% off", "all-new", "best price", "all-natural", "100% satisfaction",
    "lifetime", "urgency", "do it today!", "act fast!", "apply now", "apply online",
    "access fast", "call now", "call free", "instant access", "don’t hesitate",
    "for you", "instant", "now", "now only", "order expires", "please read",
    "take action now", "while supplies last", "one time only", "click this link",
    "click to remove", "final call", "hurry up", "immediately", "top urgent",
    "last chance", "claim your prize", "new customers only",
    "important information regarding", "financial", "money back", "dollars", "cash",
    "profit", "$$$", "big bucks", "fast cash", "extra cash", "get paid", "credit",
    "debit", "billion", "cash bonus", "best price", "bonus", "double your income",
    "free investment", "lowest interest rate", "no strings attached", "risk-free",
    "serious cash", "save money", "best rates", "unsecured credit", "pure profit",
    "best mortgage rates", "extra income", "credit card offers", "no hidden fees",
    "no hidden charges", "no hidden costs", "us dollars", "allowance", "action required",
    "why pay more?", "you are a winner", "you are selected", "very cheap",
    "avoid bankruptcy", "financial independence", "online biz opportunity", "risk-free",
    "pre-approved", "winner", "offshore", "legal", "loans", "luxury", "accept credit cards",
    "beneficiary", "claims", "claims to be legal", "shady or unethical behavior",
    "dear friend", "direct email", "bulk email", "mass email", "confidentiality",
    "cancel any time", "congratulations", "no catch", "no costs", "no gimmicks",
    "human growth hormone", "not spam", "no obligation", "babes", "cutie", "kinky", "mature",
    "viagra", "valium", "vicodin", "weight loss", "xanax", "lose weight fast",
    "stop aging", "cure baldness", "miracle", "this is not a scam", "this is not fraud",
    "not junk", "no questions asked", "internet marketing", "multi-level marketing",
    "direct marketing", "click below to access", "meet singles", "social security number",
    "search engine", "internet traffic", "password", "requires initial investment",
    "your income", "get out of debt", "marketing", "click", "click below",
    "click here to remove", "re:", "ad", "auto email removal", "email marketing",
    "email harvest", "direct marketing", "internet marketing", "internet market",
    "increase sales", "increase traffic", "marketing solutions", "mass email",
    "bulk email", "direct email", "more internet traffic", "notspam", "performance",
    "we hate spam", "will not believe your eyes", "undisclosed recipient"
])

# returns the count of spam keywords found in the text
def count_spam_keywords(text):
    text = text.lower()
    return sum(1 for word in SPAM_KEYWORDS if word in text)

# loads user provided email dataset from file if available
def load_user_provided_data():
    if os.path.exists(USER_PROVIDED_PATH):
        print("[INFO] merging user-provided email dataset...")
        try:
            user_df = pd.read_csv(
                USER_PROVIDED_PATH,
                dtype=str,
                names=["email_text", "email_type", "email_label"]
            )
            print(f"[SUCCESS] loaded {len(user_df)} user-provided emails.")
            return user_df
        except Exception as e:
            print(f"[ERROR] failed to load user-provided emails: {e}")
            return pd.DataFrame(columns=["email_text", "email_type", "email_label"])
    return pd.DataFrame(columns=["email_text", "email_type", "email_label"])

# loads master dataset from internal email database first, merges user-provided data,
# extracts urls and spam keywords, and adjusts labels based on url risk; keywords are counted but no boost is applied
def load_data():
    if not os.path.exists(DATASET_PATH):
        print("[ERROR] master_email_dataset.csv not found.")
        return None
    try:
        df = pd.read_csv(DATASET_PATH, dtype=str)
        df = df[df["email_label"].isin(["0", "1", "2"])]
        print(f"[SUCCESS] loaded {len(df)} emails from master_email_dataset.csv")
    except Exception as e:
        print(f"[ERROR] failed to load dataset: {e}")
        return None
    user_df = load_user_provided_data()
    df = pd.concat([df, user_df], ignore_index=True).drop_duplicates(subset=["email_text"])
    if "email_text" not in df.columns or "email_label" not in df.columns:
        print("[ERROR] required columns 'email_text' and 'email_label' not found.")
        return None
    df["email_text"] = df["email_text"].fillna("")
    df["email_label"] = pd.to_numeric(df["email_label"], errors="coerce")
    df["email_label"] = df["email_label"].fillna(0).astype(int)
    df["urls"] = df["email_text"].apply(extract_urls)
    df["url_risk"] = df["urls"].apply(lambda urls: check_urls(urls) if urls else 0)
    df.loc[df["url_risk"] == 2, "email_label"] = 2
    df["spam_keyword_count"] = df["email_text"].apply(count_spam_keywords)
    df["spam_boost"] = 1
    return df

# trains email classifier models using tfidf and bag of words approaches and saves them
def train_classifier():
    print("[INFO] starting email classifier training...")
    df = load_data()
    if df is None or df.empty:
        print("[ERROR] dataset is empty. skipping training.")
        return
    unique_labels = np.unique(df["email_label"])
    missing_labels = set([0, 1, 2]) - set(unique_labels)
    if missing_labels:
        missing_data = pd.DataFrame({
            "email_text": ["placeholder email"] * len(missing_labels),
            "email_label": list(missing_labels)
        })
        df = pd.concat([df, missing_data], ignore_index=True)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=df["email_label"])
    cw_dict = {lbl: wt * (2 if lbl == 2 else 1) for lbl, wt in zip([0, 1, 2], class_weights)}
    print("[INFO] training tf-idf model...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(df["email_text"])
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, df["email_label"], sample_weight=[cw_dict[label] for label in df["email_label"]])
    print("[SUCCESS] finished training tf-idf model.")
    print("[INFO] training bag of words model...")
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(df["email_text"])
    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, df["email_label"], sample_weight=[cw_dict[label] for label in df["email_label"]])
    print("[SUCCESS] finished training bag of words model.")
    joblib.dump(nb_tfidf, MODEL_TFIDF)
    joblib.dump(tfidf_vectorizer, VECTORIZER_TFIDF)
    joblib.dump(nb_bow, MODEL_BOW)
    joblib.dump(bow_vectorizer, VECTORIZER_BOW)
    print("[SUCCESS] finished email classifier training. models saved.")

if __name__ == "__main__":
    train_classifier()
