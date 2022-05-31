from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__, template_folder='template')
model = load('passengerSurvival.pkl')
condition = ['Dead', 'Alive']

# Set a post method to yield the predictions on web page
@app.route('/')
def welcome():
    return render_template('app.html')


# ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
# 'Embarked_S', 'Sex_male', 'IsMinor']


# GET REQUEST
@app.route('/', methods=['POST'])
def predict_survival():
    print(request.form['age'])
    age = request.form['age']
    print(request.form['TravelAlone'])
    travelalone = request.form['TravelAlone']
    pclass_1 = request.form['PClass_1']
    pclass_2 = request.form['PClass_2']
    embarked_c = request.form['Embarked_C']
    embarked_s = request.form['Embarked_S']
    sex_male = request.form['Sex_Male']
    isminor = request.form['IsMinor']
    # print(json.loads(survival))
    prediction = model.predict([[age, travelalone, pclass_1, pclass_2, embarked_c, embarked_s, sex_male, isminor]])
    return "<h1>Passenger is {} </h1>".format(condition[int(prediction)])

  #  return "the predicted value is {}".format(condition[int(prediction)])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
