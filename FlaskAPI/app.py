import flask
from flask import Flask, jsonify, request
import json
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

def load_models():
    file_name = "models/selected_model_smote.sav"
    with open(file_name, 'rb') as pickled:
        model = pickle.load(pickled)
    return model


@app.route('/predict', methods=['GET'])
def predict():
    
    data = pd.read_csv('../Data/sample_to_train.csv')
    
    # stub input features
    # x = np.array(data.drop('TARGET', axis=1).iloc[0]).reshape(1,-1)
    id = int(request.args['client_id'])
    x_in = np.array(data.drop('TARGET', axis=1)[data.SK_ID_CURR == id]).reshape(1,-1)
    
    # load model
    model = load_models()
    prediction = model.predict_proba(x_in)[0][1]
    response = json.dumps(float(prediction))
    return response, 200

if __name__ == '__main__':

    application.run(debug=True)