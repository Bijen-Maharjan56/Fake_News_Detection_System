from flask import Flask, request, jsonify, render_template
#Flask - creates the server
#request - reads data coming from the frontend
#jsonify - sends data back to the frontend as JSON
#render_template - serves you HTML file

from flask_cors import CORS
# Allows HTML frontend to communicate with Flask
#Without this, the browser blocks the connection
import model 
#This import gives you access to model.train() and model.predict() defined in model.py

#Create the FLASK App
app = Flask(__name__) #This tells flask to look for templates/ and static/
CORS(app)

#Train model once when server starts
model.train() #calls train() in model.py

#Route 1: Serve the HTML Page
@app.route('/')
def home():
    return render_template('index.html')
#Serves templates/index.html when user opens http://localhost:5000

#Route 2: Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    incoming = request.get_json()
    #Reads the JSON body sent by fetch() in script.js

    news = incoming. get('news', '')

    if not news.strip():
        return jsonify({'error': 'Please enter news content.'}), 400
    #400 = Bad Request

    result = model.predict(news)
    #Calls predict() in model.py
    #Returns a plain Python dict

    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)
    #jsonify() converts the dict to JSON
    #Flask sends it back to script.js

# Start server
if __name__ == '__main__':
    app.run(debug=True)