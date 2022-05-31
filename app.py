from flask import Flask, request
from joblib import load
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = load('passengerSurvival.pkl')
condition = ['Dead', 'Alive']

# GET REQUEST
@app.route('/')
def welcome():
    return "Welcome All"


['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
 'Embarked_S', 'Sex_male', 'IsMinor']


# GET REQUEST
@app.route('/predict')
def predict_survival():
    # survival = request.args.get('PassengerId')
    # print(survival)
    prediction = model.predict([[18, 1, 1, 0, 0, 0, 0, 1]])

    return "the predicted value is {}".format(condition[int(prediction)])


if __name__ == '__main__':
    app.run()
