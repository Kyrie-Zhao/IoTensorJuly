import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from feature import datasetGen
import math
import joblib
import pickle
import sys
filename = "saved_model/2060gpu_GBDT_deg.pkl"
"""gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=1
                                 , min_samples_split=2, min_samples_leaf=2, max_depth=6
                                 , init=None, random_state=None, max_features=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False
                                 )
dataset_Train_feature,dataset_Train_label, dataset_Test_feature,dataset_Test_label = datasetGen()                    
train_feat = np.array(dataset_Train_feature)
train_id = np.array(dataset_Train_label).ravel()
test_feat = np.array(dataset_Test_feature)
test_id = np.array(dataset_Test_label).ravel()

#print(train_feat.shape, train_id.shape, test_feat.shape, test_id.shape)
gbdt.fit(train_feat, train_id)

#model = SelectFromModel(gbdt, prefit=True)
print(gbdt.feature_importances_)
#pred = gbdt.predict(test_feat)
#Saved model
pickle.dump(gbdt, open(filename, 'wb'))
sys.exit(1)"""
#load trained model
gbdt = pickle.load(open(filename, 'rb'))

dataset_Test_feature,dataset_Test_label = datasetGen() 
test_feat = np.array(dataset_Test_feature)
test_id = np.array(dataset_Test_label).ravel()
pred = gbdt.predict(test_feat)
total_err = 0
print(test_feat.shape)
f_acc = 0
t_acc = 0
tw_acc = 0
for i in range(pred.shape[0]):
    if (pred[i]<=(test_id[i]*1.05)) and (pred[i]>=(test_id[i]*0.95)):
        f_acc += 1
    if (pred[i]<=(test_id[i]*1.1)) and (pred[i]>=(test_id[i]*0.9)):
        t_acc += 1
    if (pred[i]<=(test_id[i]*1.2)) and (pred[i]>=(test_id[i]*0.8)):
        tw_acc += 1
    #print(pred[i], test_id[i])
    #mae 
    err = (pred[i] - test_id[i]) 
    #rmse
    #err = (pred[i] - test_id[i])*(pred[i] - test_id[i])

    total_err += err 
print("5 acc: {}   10 acc: {}   20 acc: {}".format(f_acc/pred.shape[0],t_acc/pred.shape[0],tw_acc/pred.shape[0]))
print(list(test_id))
print(list(pred))
#mae = math.sqrt(total_err)
#print("mae {}".format(total_err / pred.shape[0]))


#rmse = math.sqrt(total_err/pred.shape[0])
#print("GBDT rmse {}".format(rmse))


