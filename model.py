import re
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Globals 
word_vectorizer = None
model           = None
FLAG_FILE       = "trained.flag"

DATASET_PATH    = "datasets/final_dataset.csv"
# Political keywords used to identify political news
POLITICAL_KEYWORDS = [
    "government", "minister", "prime minister", "president",
    "parliament", "election", "vote", "cabinet",
    "constitution", "politician", "party", "campaign",
    "policy", "law", "supreme court", "speaker",
    "province", "federal", "mayor", "ward",
    "congress", "communist", "maoist", "uml",
    "nc", "rpp", "parliamentary", "governmental",
    "ministry", "commission", "governor", "opposition",
    "coalition", "democracy", "rsp", "balen", "sudan gurung", "rabi lamechane", "sunita dongol",
    "protest", "disputes", "harka sampang"
]

# Text cleaning 
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'\[.*?\]', '', text)                 # remove [tags]
    text = re.sub(r'<.*?>', '', text)                   # remove HTML
    # text = re.sub(r'\d+', '', text)                     # remove digits
    text = re.sub(r'[^\w\s]', ' ', text)                # punctuation → space
    text = re.sub(r'\s+', ' ', text).strip()            # collapse whitespace
    return text


def build_text(row) -> str:
    source   = str(row.get('source', ''))
    title    = str(row.get('title',   '')).strip()
    body     = str(row.get('body',    '')).strip()
    has_body = int(row.get('has_body', 0))

    if source == 'SyntheticNepaliFactCheck':
        # Use title only, body contains contradictory denial language
        return title

    if has_body == 1 and body and body.lower() != 'nan':
        # Full article: title + body 
        return title + " " + body
    else:
        
        return str(row.get('content', '')).strip()

def is_political(text: str) -> bool:
    text = clean_text(text)

    matches = 0

    for keyword in POLITICAL_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            matches += 1

    # Require at least 2 political keywords
    return matches >= 2

#  Training 
def train():
    global word_vectorizer, model

    # Load dataset 
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            f"Place final_dataset.csv inside your datasets/ folder."
        )

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Total rows: {len(df)}")
    print(f"  Label balance: {df['label'].value_counts().to_dict()}")

  
    train_df = df[df['split'] == 'train'].copy()
    test_df  = df[df['split'] == 'test'].copy()

    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Build text inputs 
    X_train_raw = train_df.apply(build_text, axis=1).apply(clean_text)
    X_test_raw  = test_df.apply(build_text, axis=1).apply(clean_text)

    # predict() returns "REAL" when prediction==1 and "FAKE" when prediction==0.
    # So we must flip: (1 - label) converts 0→1 (real) and 1→0 (fake).

    Y_train = 1 - train_df["label"]
    Y_test  = 1 - test_df["label"]
    print(f"  After label flip → train REAL: {(Y_train==1).sum()}, FAKE: {(Y_train==0).sum()}")

    # Word-level TF-IDF
   
    word_vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words=None,
    ngram_range=(1,3),
    max_features=120000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

    # Character-level TF-IDF 
    # Captures writing style (punctuation density, suffixes, phrasing patterns)
    # Useful because fake vs real news often differ in style, not just vocabulary
    char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 6),
    max_features=60000,
    min_df=2,
    sublinear_tf=True
    )

    combined = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])

    print("Fitting vectorizers and training model...")
    X_train_feat = combined.fit_transform(X_train_raw)
    X_test_feat  = combined.transform(X_test_raw)

    # Calibrated LinearSVC 
    base_clf = LinearSVC(
    C=1.0,
    class_weight='balanced',
    max_iter=5000,
    random_state=42
    )
    clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5 )
    clf.fit(X_train_feat, Y_train)

    model = {
        'combined': combined,
        'clf':      clf
    }

    # Evaluate (onr time run only) 
    first_time = not os.path.exists(FLAG_FILE)
    if first_time:
        print("\nFirst run — evaluating on held-out test split...")
        y_pred = clf.predict(X_test_feat)

        print(f"  Accuracy : {accuracy_score(Y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(Y_test, y_pred):.4f}")
        print(f"  Recall   : {recall_score(Y_test, y_pred):.4f}")
        print(f"  F1 Score : {f1_score(Y_test, y_pred):.4f}")
        print()
        print(classification_report(Y_test, y_pred, target_names=["FAKE", "REAL"]))

        cm = confusion_matrix(Y_test, y_pred)
        os.makedirs("static/images", exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["FAKE", "REAL"],
                    yticklabels=["FAKE", "REAL"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix — LinearSVC + TF-IDF (final_dataset.csv)")
        plt.tight_layout()
        plt.savefig("static/images/confusion_matrix.png", dpi=150)
        plt.close()
        print("Confusion matrix saved to static/images/confusion_matrix.png")

        with open(FLAG_FILE, 'w') as f:
            f.write("trained")
    else:
        print("  Already trained. Delete trained.flag to force re-evaluation.")

    print("Model ready.")
    # Save trained objects
    joblib.dump(model, "model.pkl")
    joblib.dump(word_vectorizer, "word_vectorizer.pkl")

    print("Model saved successfully.")

#  Prediction
def predict(news: str) -> dict:
    cleaned = clean_text(news)

    # Check whether the news is political
    if not is_political(cleaned):
        return {
            "error": "This system only detects political news."
        }

    combined = model['combined']
    clf      = model['clf']

    user_word_tfidf = word_vectorizer.transform([cleaned])

    if user_word_tfidf.nnz == 0:
        return {'error': 'No recognizable words found. Please enter more content.'}

    user_features = combined.transform([cleaned])

    prediction = clf.predict(user_features)[0]
    prob = clf.predict_proba(user_features)[0]

    confidence = round(float(np.max(prob)) * 100, 2)

    print("Prediction:", prediction)
    print("Probabilities:", prob)
    print("Confidence:", confidence)

    # Cap at 95% honest confidence instead of inflated 99%
    
    label = "REAL" if prediction == 1 else "FAKE"

    # Top influential words
    feature_names = word_vectorizer.get_feature_names_out()
    weights       = clf.calibrated_classifiers_[0].estimator.coef_[0]
    indices       = user_word_tfidf.nonzero()[1]

    word_contributions = []
    for i in indices:
        contribution = float(user_word_tfidf[0, i] * weights[i])
        word_contributions.append({'word': feature_names[i], 'score': round(contribution, 4)})

    word_contributions = sorted(
        word_contributions, key=lambda x: abs(x['score']), reverse=True
    )[:5]

    return {
        'prediction': label,
        'confidence': confidence,
        'top_words':  word_contributions
    }

def load_model():
    global model
    global word_vectorizer

    model = joblib.load("model.pkl")
    word_vectorizer = joblib.load("word_vectorizer.pkl")

    print("Saved model loaded.")