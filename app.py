from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import model 

app = Flask(__name__) 
CORS(app)


model.load_model()

#Route 1: Serve the HTML Page
@app.route('/')
def home():
    return render_template('home.html')
#Serves templates/home.html when user opens http://localhost:5000

#Route 2: Prediction Interface Page
@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

#Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    incoming = request.get_json()
   

    news = incoming. get('news', '')

    if not news.strip():
        return jsonify({'error': 'Please enter news content.'}), 400
    #400 = Bad Request

    result = model.predict(news)


    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)
   

# Start server
if __name__ == '__main__':
    app.run(debug=True)