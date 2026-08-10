# Fake News Detection System
A web based machine learning application that detects whether a political news article is REAL or FAKE. The system is built using Flask and Linear SVC with TF-IDF feature extraction. 

# Features
- Detects fake and real political news.
- Accepts news headlines and full articls.
- Displays confidence score.
- Shows influential words behind the prediction.
- Rejects non-political news.

# How to Run

1. Clone the repository.
2. Install the required packages. 
    pip install -r requirements.txt
3. Start the application.
    python app.py
4. Open browser and visit: 
    http://127.0.0.1:5000

# Note: 
The repository already includes the trained model (model.pkl and word_vectorizer.pkl). You need to run "python train_model.py" only if you modify the dataset and want to retrain the mode.
