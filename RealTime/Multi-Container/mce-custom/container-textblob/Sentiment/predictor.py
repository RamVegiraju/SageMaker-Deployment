from flask import Flask
import flask
import os
import json
import logging
from textblob import TextBlob



# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = TextBlob is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    
    #Process input
    input_json = flask.request.get_json()
    resp = input_json['input']
    
    #Sentiment Analysis
    sent = TextBlob(resp).sentiment
    polarity, subjectivity = sent[0], sent[1]
    print(polarity)
    print(type(polarity))
    print(subjectivity)
    print(type(subjectivity))
    result = {"Polarity": polarity, "Subjectivity": subjectivity}
    print(result)
    print(type(result))

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')