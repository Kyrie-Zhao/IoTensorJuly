import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from feature import datasetGen
import math
import joblib
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
dataset_Train_feature,dataset_Train_label, dataset_Test_feature,dataset_Test_label = datasetGen()                    
train_feat = np.array(dataset_Train_feature)
train_id = np.array(dataset_Train_label).ravel()
test_feat = np.array(dataset_Test_feature)
test_id = np.array(dataset_Test_label).ravel()

#print(train_feat.shape, train_id.shape, test_feat.shape, test_id.shape)
regressor.fit(train_feat, train_id)
pred = regressor.predict(test_feat)
#Saved model
#joblib.dump(pred, 'saved_model/2060gpu_RF.pkl')
total_err = 0

#print(train_feat.shape)
for i in range(pred.shape[0]):
    #print(pred[i], test_id[i])
    #mae 
    err = (pred[i] - test_id[i]) 
    #rmse
    #err = (pred[i] - test_id[i])*(pred[i] - test_id[i])
    
    total_err += err 
mae = math.sqrt(total_err)
#rmse = math.sqrt(total_err/pred.shape[0])
print("mae {}".format(total_err / pred.shape[0]))
print("RF rmse {}".format(rmse))


