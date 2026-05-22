#This file's job: load data, train model, make predections
#IMPORTS

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#These are created once and shared across all functions in this file
#app.py can access them after calling train()
vectorizer = None
model = None

#We use a file to remember if training has been done before
#So eveluation and graph only runs on the very first launch
FLAG_FILE = "trained.flag" 

def train():
    global vectorizer, model
    #Phase 1: Load and prepare data
    fake = pd.read_csv("datasets/Fake.csv")
    true = pd.read_csv("datasets/Real.csv")
    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], ignore_index=True)

    X = data['title'].fillna('') + " " + data['content'].fillna('') #Combining title and content
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Phase 2: Train the model
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, Y_train)

    #Phase 3: Evaluate and Graph
    first_time = not os.path.exists(FLAG_FILE)
    #os.path.exists() checks if the flag file is on disk
    #If the file does not exists, this is the first time
    #If the file exists, already ran before, skip this

    if first_time: 
        print("\n First run: Evaluating model")
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        cm = confusion_matrix(Y_test, y_pred)

        print("Logistic Regression:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\n Classification Report:")
        print(classification_report(Y_test, y_pred, target_names=["FAKE", "REAL"]))

        #Saving confusion matrix as an image file instead of showing a popup (which doesnot work in Flask)
        plt.figure(figsize=(6,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"]
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix - Logistic Regression + TF-IDF")
        plt.tight_layout()
        plt.savefig("static/images/confusion_matrix.png", dpi=150)
        plt.close() #frees memory after saving

        print("Confusion matrix saved to static/images/confusion_matrix.png")

        with open(FLAG_FILE, 'w') as f:
            f.write("trained")
        #Next time train() is called, os.path.exists(FLAG_FILE) will return True, so the if block is skipped
    else:
        print("Model trained already.")
    print("Model ready.")

def predict(news):
    #Combine and vectorize input
    user_text = news
    user_tfidf = vectorizer.transform([news])
    
    #Checking if any words were recognized
    if user_tfidf.nnz == 0: #nnz - number of non-zero features
        return {'error': 'No recognizable words found. Please enter more content.'}
    
    #Get prediction
    prediction = model.predict(user_tfidf)[0]
    #Returns 0 (FAKE) or 1 (REAL)

    prob = model.predict_proba(user_tfidf)[0]
    confidence = round(float(np.max(prob)) * 100, 2)
    #predict_proba returns [prob_fake, prob_real]
    #np.max picks the higher probability
    #Multiply by 100 -> precentage

    label = "REAL" if prediction == 1 else "FAKE"
    
    #Get top influential words (reasoning)
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    indices = user_tfidf.nonzero()[1]

    word_contributions = []
    for i in indices:
        word = feature_names[i]
        tfidf_value = user_tfidf[0, i]
        weight = weights[i]
        contribution = float(tfidf_value * weight)
        word_contributions.append({
            'word' : word,
            'score' : round(contribution, 4)
        })
        #contribution = tfidf_value * weight
        #Positive - word pushes prediction toward REAL
        #Negative - word pushes prediction toward FAKE

    word_contributions = sorted(
        word_contributions,
        key=lambda x: abs(x['score']),
        reverse=True
    )[:5]
    #Sort by absolute value - biggest influence first
    #[:5] takes only top 5

    # Return plain Python dict - No jsonify here
    return {
        'prediction' : label,
        'confidence' : confidence,
        'top_words' : word_contributions
    }

#app.py will call jsonify() on this dict. model.py stays completely independent of Flask