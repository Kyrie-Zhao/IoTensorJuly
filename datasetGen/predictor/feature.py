import json
import sys
import os
import numpy as np
sys.path.append("..") 
from testing_model import inference

def datasetGen():
    #testting model inference isolation
    model_bag = {"1":"inception_v3","2":"googlenet","3":"squeezenet","4":"shufflenet","5":"vgg_11","6":"vgg_13","7":"vgg_16","8":"mobilenet_v3_small","9":"alexnet","10":"mvit","11":"mnasnet","12":"resnext50_32x4d","13":"mobilenet_v3_large","14":"mobilenet_v2","15":"segnet",'16':"efficientnet_b0"}
    inverse_model_bag =  {v:k for k,v in model_bag.items()}
    
    """label_list =[]
    for _ in range(1,17):
        time = inference(model_bag[str(_)],1)
        print("Model {} inference time: {}".format(model_bag[str(_)],str(time)))
        label_list.append(time)
    label_list = [round(i,3) for i in label_list]
    sys.exit(1)"""
    label_list = [7.582, 4.595, 1.635, 4.33, 2.762, 3.659, 4.755, 3.955, 0.681, 7.453, 3.427, 5.964, 4.688, 3.542,85.724,9.183]
    #flops gflops, memoryRW mb, memory mb
    model_info = [[2850,258.67,70.63],[1510,96.36,30.03],[829.88,76.52,35.6],[150.6,51.57,20.85],[7630,632.52,62.69],[11340,706.73,99.44],[15500,746.89,109.39],[59.81,35.35,16.2],[715.54,241.86,4.19],[1120,246.96,179.95],[330.13,136.93,59.94],[4270,365.56,134.76],[227.71,106.17,50.4],[320.24,162.2,74.25],[298.56,4950,2494.85],[575.29,200.53,128.94]]

    #Training 
    filePath = '../dataset/'
    #testing dataset

    #Testing
    #filePath = '../tmpdb/'
    #filePath = '../testing/'
    #file_list = [a for a in os.listdir(filePath) if ]
    file_list = os.listdir(filePath)
    #print(file_list)
    dataset_Train_feature = []
    dataset_Test_feature = []

    dataset_Train_label = []
    dataset_Test_label = []

    dataset_Train_len = len(file_list)-500
    dataset_Test_len = 500

    dataset_NewTest_len = len(file_list)


    #new testing
    test = 1
    if test:
        
        for i in range(0,dataset_NewTest_len):
            dataset_Test_feature_tmp = []
            dataset_Test_label_tmp = []
            dataID = file_list[i].split("-")[0]
            #print(dataID)
            
            with open(filePath+file_list[i],'r') as f:
                x = json.load(f)
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cpu-cycles'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['instructions'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cache-references'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cache-misses'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-loads'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-load-misses'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-store-misses'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-load-misses'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-stores'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['GPU_Util'])
                dataset_Test_feature_tmp.append(x[dataID]["global"][0]['GPU_Mem'])

                modelID = int(inverse_model_bag[x[dataID]["inference-model-name"]])
                model_flops = model_info[modelID-1][0]
                model_memoryRW = model_info[modelID-1][1]
                model_memory = model_info[modelID-1][2]

                dataset_Test_feature_tmp.append(model_flops)
                dataset_Test_feature_tmp.append(model_memoryRW)
                dataset_Test_feature_tmp.append(model_memory)
                
                degradation = round(x[dataID]["inference-model-latency"]/label_list[modelID-1],2) #round(x[dataID]["inference-model-latency"],2)
                dataset_Test_label_tmp.append(degradation)
                f.close()
            dataset_Test_feature.append(dataset_Test_feature_tmp)
            dataset_Test_label.append(dataset_Test_label_tmp)
            #print(dataset_Test_feature)
            #print(dataset_Test_label)
        return dataset_Test_feature,dataset_Test_label


    for i in range(0,dataset_Train_len):
        dataset_Train_feature_tmp = []
        dataset_Train_label_tmp = []
        dataID = file_list[i].split("-")[0]
        #print(dataID)
        
        with open(filePath+file_list[i],'r') as f:
            x = json.load(f)
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['cpu-cycles'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['instructions'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['cache-references'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['cache-misses'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['LLC-loads'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['LLC-load-misses'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['LLC-store-misses'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-load-misses'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-stores'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['GPU_Util'])
            dataset_Train_feature_tmp.append(x[dataID]["global"][0]['GPU_Mem'])

            modelID = int(inverse_model_bag[x[dataID]["inference-model-name"]])
            model_flops = model_info[modelID-1][0]
            model_memoryRW = model_info[modelID-1][1]
            model_memory = model_info[modelID-1][2]

            dataset_Train_feature_tmp.append(model_flops)
            dataset_Train_feature_tmp.append(model_memoryRW)
            dataset_Train_feature_tmp.append(model_memory)
            
            degradation = round(x[dataID]["inference-model-latency"]/label_list[modelID-1],2)#round(x[dataID]["inference-model-latency"],2)#
            dataset_Train_label_tmp.append(degradation)
            f.close()
        dataset_Train_feature.append(dataset_Train_feature_tmp)
        dataset_Train_label.append(dataset_Train_label_tmp)
        #print(len(dataset_Train_feature))
        #print(dataset_Train_label)

    for i in range(dataset_Train_len,dataset_Train_len+500):
        dataset_Test_feature_tmp = []
        dataset_Test_label_tmp = []
        dataID = file_list[i].split("-")[0]
        #print(dataID)
        
        with open(filePath+file_list[i],'r') as f:
            x = json.load(f)
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cpu-cycles'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['instructions'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cache-references'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['cache-misses'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-loads'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-load-misses'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['LLC-store-misses'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-load-misses'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['L1-dcache-stores'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['GPU_Util'])
            dataset_Test_feature_tmp.append(x[dataID]["global"][0]['GPU_Mem'])

            modelID = int(inverse_model_bag[x[dataID]["inference-model-name"]])
            model_flops = model_info[modelID-1][0]
            model_memoryRW = model_info[modelID-1][1]
            model_memory = model_info[modelID-1][2]

            dataset_Test_feature_tmp.append(model_flops)
            dataset_Test_feature_tmp.append(model_memoryRW)
            dataset_Test_feature_tmp.append(model_memory)
            
            degradation = round(x[dataID]["inference-model-latency"]/label_list[modelID-1],2) #round(x[dataID]["inference-model-latency"],2)
            dataset_Test_label_tmp.append(degradation)
            f.close()
        dataset_Test_feature.append(dataset_Test_feature_tmp)
        dataset_Test_label.append(dataset_Test_label_tmp)
        #print(dataset_Test_feature)
        #print(dataset_Test_label)
    return dataset_Train_feature,dataset_Train_label, dataset_Test_feature,dataset_Test_label
              
