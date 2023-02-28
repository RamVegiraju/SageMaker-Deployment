from flask import Flask
import flask
import os
import json
import logging
import xgboost as xgb
from io import StringIO
import pandas as pd

xgb_reg = xgb.Booster()
xgb_reg.load_model("model.json")

# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the pre-trained xgboost regressor was loaded correctly
    health = xgb_reg is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )
    
    print(data)
    print(type(data))
    preds = xgb_reg.predict(xgb.DMatrix(data))
    print(preds)
    
    out = StringIO()
    pd.DataFrame({"results": preds}).to_csv(out, header=False, index=False)

    result = out.getvalue().rstrip(
        "\n"
    )
    
    return flask.Response(response=result, status=200, mimetype='text/csv')
