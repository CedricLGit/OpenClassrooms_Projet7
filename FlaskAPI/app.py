import flask
from flask import Flask, jsonify, request
import json
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

def load_models():
    file_name = "Data_API/model_smote_v3.sav"
    with open(file_name, 'rb') as pickled:
        model = pickle.load(pickled)
    return model


@app.route('/predict', methods=['GET'])
def predict():
    
    # retrieve input from request of dashboard
    data_input = request.get_json()
    x = data_input['input']
    x_in = np.array(x)
    
    # load model
    model = load_models()
    prediction = model.predict_proba(x_in)[0][1]
    response = json.dumps(round(float(prediction),3))
    return response, 200

if __name__ == '__main__':

    application.run(debug=True)