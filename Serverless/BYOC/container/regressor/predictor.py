from __future__ import print_function
import io
import json
import os
import pickle
import signal
import sys
import traceback
from flask import Flask
import flask
import pandas as pd

#load model
with open("model.pkl", "rb") as inp:
    lgbm_predictor = pickle.load(inp)

# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = lgbm_predictor is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    
    #Process input
    input_json = flask.request.get_json()
    sampInp = input_json['input']
    
    #NER
    resp = lgbm_predictor.predict(sampInp).tolist()
    print(resp)
    print(type(resp))

    # Transform predictions to JSON
    result = {
        'output': resp
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')