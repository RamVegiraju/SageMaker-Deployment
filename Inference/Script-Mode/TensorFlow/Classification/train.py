#Necessary Imports
import argparse, os
import boto3
import numpy as np
import pandas as pd
import sagemaker
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K


if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
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
    X = df.drop(['Species'],axis=1)
    y = df['Species']
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler() #scaling X data before model training
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    ###########
    #Model Building
    ###########
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_shape=[4,]))
    model.add(Dropout(.3))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',metrics=['accuracy'])
    early_stop = EarlyStopping(patience=10)
    model.fit(x=scaled_X_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(scaled_X_test, y_test), verbose=1 ,callbacks=[early_stop])
    
    #Storing model artifacts
    model.save(os.path.join(sm_model_dir, '000000001'), 'my_model.h5')