# Necessary imports
import numpy as np
import pandas as pd
import xgboost as xg
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.datasets import load_boston
from feature import datasetGen
filename = "saved_model/2060gpu_xgb_deg.pkl"
# Load the data
#dataset = pd.read_csv("boston_house.csv")

"""
dataset_Train_feature,dataset_Train_label, dataset_Test_feature,dataset_Test_label = datasetGen()                    
train_X = np.array(dataset_Train_feature)
train_y = np.array(dataset_Train_label).ravel()
test_X = np.array(dataset_Test_feature)
test_y = np.array(dataset_Test_label).ravel()
# Instantiation
xgb_r = xg.XGBRegressor(objective ='reg:squarederror',max_depth = 8,
                  n_estimators = 300, seed = 123)

# Fitting the model
xgb_r.fit(train_X, train_y)
#Saved model
pickle.dump(xgb_r, open(filename, 'wb'))
sys.exit(1)"""
#load trained model
xgb_r = pickle.load(open(filename, 'rb'))

dataset_Test_feature,dataset_Test_label = datasetGen() 
test_feat = np.array(dataset_Test_feature)
test_y = np.array(dataset_Test_label).ravel()
pred = xgb_r.predict(test_feat)
#load model
gbdt = pickle.load(open(filename, 'rb'))

# Predict the model
#pred = xgb_r.predict(test_X)

# save predictor
#joblib.dump(pred, 'saved_model/2060gpu_xgb.pkl')
# RMSE Computation
rmse = np.sqrt(MSE(test_y, pred))
print("XGB RMSE : % f" %(rmse))
print(MAE(test_y, pred))