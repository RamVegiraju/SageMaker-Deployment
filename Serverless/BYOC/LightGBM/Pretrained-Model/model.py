import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import lightgbm as lgb


petrolDF = pd.read_csv("petrol_consumption.csv")
X = petrolDF.drop('Petrol_Consumption', axis = 1)
y = petrolDF["Petrol_Consumption"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = lgb.Dataset(X, y)
test_dataset = lgb.Dataset(X_test, y_test)


#Model Creation
booster = lgb.train({"objective": "regression"},
                    train_set=train_dataset,
                    num_boost_round=10)

pickle.dump(booster, open("model.pkl", "wb"))

# load model & inference
model = pickle.load(open("model.pkl", "rb"))
print(model.predict(X_test[:1])[0])