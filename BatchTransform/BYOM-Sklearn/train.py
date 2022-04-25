import argparse, os
import boto3
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
import pickle
from io import StringIO

if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Hyperparamaters
    parser.add_argument('--estimators', type=int, default=15)
    
    #sm_model_dir: model artifacts stored here after training
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    estimators     = args.estimators
    model_dir  = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir   = args.train
    
    ############
    #Reading in data
    ############
    df = pd.read_csv(training_dir + '/train.csv',sep=',')
    
    ############
    #Preprocessing data
    ############
    X = df.drop('Petrol_Consumption', axis = 1)
    y = df['Petrol_Consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    ###########
    #Model Building
    ###########
    regressor = RandomForestRegressor(n_estimators=estimators)
    regressor.fit(X_train, y_train)
    
    ###########
    #Save the Model
    ###########
    joblib.dump(regressor, os.path.join(args.sm_model_dir, "model.joblib"))
    

###########
#Model Serving
###########
    
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(input_data, content_type):
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        df = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)
