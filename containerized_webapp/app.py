from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import train

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        # message = train.preprocess(message)
        # data = [message]
        # data = cv.transform(data)
        # print("transformed", data)
        my_prediction = (clf.predict(cv.transform([train.preprocess(message)])))
        print(my_prediction)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':


    ##load vectorizer and model
    with open('model/logistic_clf.pkl', 'rb') as f:
        cv, clf = pickle.load(f)

    app.run(host='0.0.0.0',port=5000)
