import json
import sys
import os
import numpy as np
sys.path.append("..") 
from testing_model import inference
def location_square_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if n != len(lst_2):
            return False
        for i in range(n):	
            lst[lst_1.index(lst_2[i])] = i

    s = 0
    for i in range(n):
        s += (lst[i]-i) ** 2
    s /= n
    return s


model_bag = {"1":"inception_v3","2":"googlenet","3":"squeezenet","4":"shufflenet","5":"vgg_11","6":"vgg_13","7":"vgg_16","8":"mobilenet_v3_small","9":"alexnet","10":"mvit","11":"mnasnet","12":"resnext50_32x4d","13":"mobilenet_v3_large","14":"mobilenet_v2","15":"segnet",'16':"efficientnet_b0"}
label_list = [7.582, 4.595, 1.635, 4.33, 2.762, 3.659, 4.755, 3.955, 0.681, 7.453, 3.427, 5.964, 4.688, 3.542,85.724,9.183]

filePath = 'dataset/'
file_list = os.listdir(filePath)
file_len = len(file_list)

model_1 = {}
model_2 = {}
model_3 = {}
model_4 = {}
model_5 = {}
model_6 = {}
model_7 = {}
model_8 = {}
model_9 = {}
model_10 = {}
model_11 = {}
model_12 = {}
model_13 = {}
model_14 = {}
inverse_model_bag =  {v:k for k,v in model_bag.items()}
for i in range(0,file_len):
    inferenceModelID = file_list[i].split("-")[2].split(".")[0]
    dataID = file_list[i].split("-")[0]
    #print(dataID)
    #print(inferenceModelID)
    
    with open(filePath+file_list[i],'r') as f:
        x = json.load(f)
        #print(x)
        #sys.exit(1)
        modelID = int(inverse_model_bag[x[dataID]["inference-model-name"]])
        degradation = round(x[dataID]["inference-model-latency"]/label_list[modelID-1],2) #round(x[dataID]["inference-model-latency"],2)
        f.close()
    dataID = file_list[i].split("-")[1]
 
    if (inferenceModelID == "1"):
        model_1[dataID] = degradation
    if (inferenceModelID == "2"):
        model_2[dataID] = degradation
    if (inferenceModelID == "3"):
        model_3[dataID] = degradation
    if (inferenceModelID == "4"):
        model_4[dataID] = degradation
    if (inferenceModelID == "5"):
        model_5[dataID] = degradation
    if (inferenceModelID == "6"):
        model_6[dataID] = degradation
    if (inferenceModelID == "7"):
        model_7[dataID] = degradation
    if (inferenceModelID == "8"):
        model_8[dataID] = degradation
    if (inferenceModelID == "9"):
        model_9[dataID] = degradation
    if (inferenceModelID == "10"):
        model_10[dataID] = degradation
    if (inferenceModelID == "11"):
        model_11[dataID] = degradation
    if (inferenceModelID == "12"):
        model_12[dataID] = degradation
    if (inferenceModelID == "13"):
        model_13[dataID] = degradation
    if (inferenceModelID == "14"):
        model_14[dataID] = degradation

#d = {'lilee':25, 'wangyan':21, 'liqun':32, 'age':19}
model_1 = sorted(model_1.items(), key=lambda item:item[1])
rank_1 = [a[0] for a in model_1]

model_2 = sorted(model_2.items(), key=lambda item:item[1])
rank_2 = [a[0] for a in model_2]
model_3 = sorted(model_3.items(), key=lambda item:item[1])
rank_3 = [a[0] for a in model_3]
model_4 = sorted(model_4.items(), key=lambda item:item[1])
rank_4 = [a[0] for a in model_4]
model_5 = sorted(model_5.items(), key=lambda item:item[1])
rank_5 = [a[0] for a in model_5]
model_6 = sorted(model_6.items(), key=lambda item:item[1])
rank_6 = [a[0] for a in model_6]
model_7 = sorted(model_7.items(), key=lambda item:item[1])
rank_7 = [a[0] for a in model_7]
model_8 = sorted(model_8.items(), key=lambda item:item[1])
rank_8 = [a[0] for a in model_8]
model_9 = sorted(model_9.items(), key=lambda item:item[1])
rank_9 = [a[0] for a in model_9]
model_10 = sorted(model_10.items(), key=lambda item:item[1])
rank_10 = [a[0] for a in model_10]
model_11 = sorted(model_11.items(), key=lambda item:item[1])
rank_11 = [a[0] for a in model_11]
model_12 = sorted(model_12.items(), key=lambda item:item[1])
rank_12 = [a[0] for a in model_12]
model_13 = sorted(model_13.items(), key=lambda item:item[1])
rank_13 = [a[0] for a in model_13]
model_14 = sorted(model_14.items(), key=lambda item:item[1])
rank_14 = [a[0] for a in model_14]


model_1_json = json.dumps(model_1)
f = open("draw/model_1.json","w")
f.write(model_1_json)

model_2_json = json.dumps(model_2)
f = open("draw/model_2.json","w")
f.write(model_2_json)

model_3_json = json.dumps(model_3)
f = open("draw/model_3.json","w")
f.write(model_3_json)

model_4_json = json.dumps(model_4)
f = open("draw/model_4.json","w")
f.write(model_4_json)

model_5_json = json.dumps(model_5)
f = open("draw/model_5.json","w")
f.write(model_5_json)

model_6_json = json.dumps(model_6)
f = open("draw/model_6.json","w")
f.write(model_6_json)

model_7_json = json.dumps(model_7)
f = open("draw/model_7.json","w")
f.write(model_7_json)

model_8_json = json.dumps(model_8)
f = open("draw/model_8.json","w")
f.write(model_8_json)

model_9_json = json.dumps(model_9)
f = open("draw/model_9.json","w")
f.write(model_9_json)

model_10_json = json.dumps(model_10)
f = open("draw/model_10.json","w")
f.write(model_10_json)

model_11_json = json.dumps(model_11)
f = open("draw/model_11.json","w")
f.write(model_11_json)

model_12_json = json.dumps(model_12)
f = open("draw/model_12.json","w")
f.write(model_12_json)

model_13_json = json.dumps(model_13)
f = open("draw/model_13.json","w")
f.write(model_13_json)

model_14_json = json.dumps(model_14)
f = open("draw/model_14.json","w")
f.write(model_14_json)
print(rank_1[:30])
print(rank_2[:30])
print(rank_3[:30])

print(location_square_deviation(rank_1[:100],rank_7[:100]))