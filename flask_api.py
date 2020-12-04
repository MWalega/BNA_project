from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
pickle_in = open('/home/maciek/Desktop/BNA_project/classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome all'

@app.route('/predict')
def bank_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = classifier([
                             [variance,
                              skewness,
                              curtosis,
                              entropy]
                              ])

    return 'The predicted value is:' + str(prediction)

if __name__ == '__main__':
    app.run()