from flask import Flask, jsonify,render_template,url_for,request
import pandas as pd 
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import train_model

app = Flask(__name__)

# Default page of the web-app
@app.route('/')
def home():
    return render_template('home.html')

# Page to notify training is done
@app.route('/train')
def train():
    # Trigger model training
    train_model.main()
    return render_template('train.html')

# Page to get input for prediction
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Page to show predictions
@app.route('/result',methods=['POST'])
def result():
    ## Load vectorizer and model
    cv, clf = load_model()
    # Get data from browser
    message = request.form['message']
    # Pass input for model to predict
    my_prediction = (clf.predict(cv.transform([train_model.preprocess(message)])))
    # Postprocess prediction for output
    gender = np.where(my_prediction[0] == 1, "Male", "Female")
    return render_template('predict.html', prediction_text="Gender is {}".format(gender))

def load_model():
    "Load countvectorizer and final tuned model"
    try:
        with open('model/final_model.pkl', 'rb') as f:
            cv, clf = pickle.load(f)
    except FileNotFoundError:
        print('Model does not exist, train model first')
    return cv, clf

if __name__ == '__main__':


    app.run(host='0.0.0.0',port=5000)
