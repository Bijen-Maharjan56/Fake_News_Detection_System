#Required Libraries
import pandas as pd #For handeling dataset
import numpy as np
from sklearn.model_selection import train_test_split #For splitting the data
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression #Logistic Regression Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Loading Datasets
fake = pd.read_csv("datasets/Fake.csv")
true = pd.read_csv("datasets/True.csv")

print(fake.head())
print(true.head())

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
#pd.concat: stacks both dataframes on top of each other

print(data.shape)
print(data.head())

# 'text' column contains the news content
X = data['text'] #features (input text)
Y = data['label'] #Labels (0 = Fake, 1 = Real)

#Splitting the data into 80% Training and 20% Testing
X_train,X_test,Y_train,Y_test = train_test_split( X,Y, test_size=0.2, random_state=42)
#random_state=42 ensures same split everytime

#Using TF-IDF to convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english') #removes common words like 'this', 'is'

#learn + convert training data
X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

#Training the Naive Bayes Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, Y_train)

#Make prediction
y_pred = model.predict(X_test_tfidf)

#Evaluate Model
#Accuracy 
accuracy = accuracy_score(Y_test, y_pred)

#Precision
precision = precision_score(Y_test,y_pred)

#Recall
recall = recall_score(Y_test,y_pred)

#F1-score
f1 = f1_score(Y_test,y_pred)

#Confusion Matrix
cm = confusion_matrix(Y_test,y_pred)

print("Logistic Regression:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, y_pred, target_names=["FAKE", "REAL"]))

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,          # Show numbers inside boxes
    fmt="d",             # Display as integers
    cmap="Blues",        # Color scheme
    xticklabels=["FAKE", "REAL"],   # Predicted labels (x-axis)
    yticklabels=["FAKE", "REAL"]    # Actual labels (y-axis)
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Logistic Regression + TF-IDF")
plt.tight_layout()
#plt.savefig("confusion_matrix.png", dpi=150)  # Save the plot
plt.show()

# ================= USER INPUT =================

print("\n--- Fake News Detection ---")

# Take input from user
heading = input("Enter News Heading (optional):\n")
content = input("Enter News Content:\n")

# Combine both
user_text = heading + " " + content

# Convert to TF-IDF
user_tfidf = vectorizer.transform([user_text])

# Predict
prediction = model.predict(user_tfidf)[0]

# Confidence score
prob = model.predict_proba(user_tfidf)[0]
confidence = np.max(prob)

# Output result
print("\nPrediction:", "REAL" if prediction == 1 else "FAKE")
print("Confidence:", round(confidence * 100, 2), "%")

# ================= EXPLANATION =================

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

indices = user_tfidf.nonzero()[1]

word_contributions = []

for i in indices:
    word = feature_names[i]
    tfidf_value = user_tfidf[0, i]
    weight = weights[i]
    contribution = tfidf_value * weight
    
    word_contributions.append((word, contribution))

# Sort by importance
word_contributions = sorted(word_contributions, key=lambda x: abs(x[1]), reverse=True)

print("\nTop Influential Words:")
for word, score in word_contributions[:5]:
    print(f"{word}: {round(score, 4)}")



# # Get probability scores
# y_prob = model.predict_proba(X_test_tfidf)

# # Get confidence for each prediction
# confidence_scores = np.max(y_prob, axis=1)

# # Show first 5 predictions with confidence
# for i in range(5):
#     print("News:", X_test.iloc[i][:100], "...")
#     print("Prediction:", "REAL" if y_pred[i] == 1 else "FAKE")
#     print("Confidence:", round(confidence_scores[i]*100, 2), "%")
#     print("-" * 50)

# feature_names = vectorizer.get_feature_names_out()

# def explain_prediction(text):
#     # Convert text to TF-IDF
#     text_tfidf = vectorizer.transform([text])
    
#     # Get prediction
#     pred = model.predict(text_tfidf)[0]
    
#     # Get probabilities
#     prob = model.predict_proba(text_tfidf)[0]
#     confidence = np.max(prob)
    
#     # Get feature weights
#     weights = model.coef_[0]
    
#     # Get non-zero indices (words present in text)
#     indices = text_tfidf.nonzero()[1]
    
#     word_contributions = []
    
#     for i in indices:
#         word = feature_names[i]
#         tfidf_value = text_tfidf[0, i]
#         weight = weights[i]
#         contribution = tfidf_value * weight
        
#         word_contributions.append((word, contribution))
    
#     # Sort by importance
#     word_contributions = sorted(word_contributions, key=lambda x: abs(x[1]), reverse=True)
    
#     # Get top 5 words
#     top_words = word_contributions[:5]
    
#     # Print results
#     print("\nNews Text:", text)
#     print("Prediction:", "REAL" if pred == 1 else "FAKE")
#     print("Confidence:", round(confidence*100, 2), "%")
    
#     print("\nTop Influential Words:")
#     for word, score in top_words:
#         print(f"{word}: {round(score, 4)}")

# user_input = input("Enter news text:\n") 
# explain_prediction(user_input)
