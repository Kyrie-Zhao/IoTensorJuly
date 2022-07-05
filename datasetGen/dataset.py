import torchvision
from testing_model import inference
import time
import os
import numpy as np
import torch
import time
import argparse
import sys
import json
import time
import sys
import subprocess
from torchstat import stat
import GPUtil
import itertools
import warnings
warnings.filterwarnings("ignore",category=UserWarning) 
def file_Process(filename):
    perf_counter = {}
    perf_counter['cpu-cycles'] = 0
    perf_counter['instructions'] = 0
    perf_counter['cache-references'] = 0
    perf_counter['cache-misses'] = 0
    perf_counter['LLC-loads'] = 0
    perf_counter['LLC-load-misses'] = 0
    perf_counter['LLC-store-misses'] = 0
    perf_counter['LLC-stores'] = 0
    perf_counter['L1-dcache-load-misses'] = 0
    perf_counter['L1-dcache-loads'] = 0
    perf_counter['L1-dcache-stores'] = 0
    perf_counter['GPU_Util'] = 0
    perf_counter['GPU_Mem'] = 0
    f = open(filename,encoding = "utf-8")
    txt_content_all = f.readlines()
    for _ in range(0,len(txt_content_all)):
        if "cpu-cycles" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            cpu_cycles = int(a[i].replace(',',""))
            perf_counter['cpu-cycles'] = cpu_cycles
            pass
        if "instructions" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            instructions = int(a[i].replace(',',""))
            perf_counter['instructions'] = instructions
            pass
        if "cache-references" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            cache_references = int(a[i].replace(',',""))
            perf_counter['cache-references'] = cache_references
            pass
        if "cache-misses" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            cache_misses = int(a[i].replace(',',""))
            perf_counter['cache-misses'] = cache_misses
            pass
        if "LLC-loads" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            LLC_loads = int(a[i].replace(',',""))
            perf_counter['LLC-loads'] = LLC_loads
            pass
        if "LLC-load-misses" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            LLC_load_misses = int(a[i].replace(',',""))
            perf_counter['LLC-load-misses'] = LLC_load_misses
            pass
        if "LLC-store-misses" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            LLC_store_misses = int(a[i].replace(',',""))
            perf_counter['LLC-store-misses'] = LLC_store_misses
            pass
        if "LLC-stores" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            LLC_stores = int(a[i].replace(',',""))
            perf_counter['LLC-stores'] = LLC_stores
            pass
        if "L1-dcache-load-misses" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            L1_dcache_load_misses = int(a[i].replace(',',""))
            perf_counter['L1-dcache-load-misses'] = L1_dcache_load_misses
            pass
        if "L1-dcache-loads" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            L1_dcache_loads = int(a[i].replace(',',""))
            perf_counter['L1-dcache-loads'] = L1_dcache_loads
            pass
        if "L1-dcache-stores" in txt_content_all[_]:
            a = txt_content_all[_].split(" ")
            for i in range(0,len(a)):
                if(a[i]!=""):
                    break
            L1_dcache_stores = int(a[i].replace(',',""))
            perf_counter['L1-dcache-stores'] = L1_dcache_stores
            pass
    f.close()

    #GPU utilizations
    gpu = GPUtil.getGPUs()[0]
    gpu_memory = round(gpu.memoryUtil,2)
    gpu_util = round(gpu.load,2)
    perf_counter['GPU_Util'] = gpu_util
    perf_counter['GPU_Mem'] = gpu_memory
    return perf_counter


def profiler(pid_list,iter_model):
    model_bag = {"1":"inception_v3","2":"googlenet","3":"squeezenet","4":"shufflenet","5":"vgg_11","6":"vgg_13","7":"vgg_16","8":"mobilenet_v3_small","9":"alexnet","10":"mvit","11":"mnasnet","12":"resnext50_32x4d","13":"mobilenet_v3_large","14":"mobilenet_v2","15":"segnet",'16':"efficientnet_b0"}
    #global resource + pid resource
    global_sys_info_list = [""]
    print("pid list {}".format(pid_list))
    
    #global sys monitor 
    os.popen('perf stat -o tmp_sys_record.txt -e cpu-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-store-misses,LLC-stores,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses -a sleep 1').read()
    sys_perf_counter = file_Process("tmp_sys_record.txt")

    #sys monitor for each process
    model_perf_sons = []
    for _ in range(0,len(pid_list)):
        workload_perf = os.popen('perf stat -o {} -p {} -e cpu-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-store-misses,LLC-stores,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses -a sleep 1'.format(str(_)+".txt",pid_list[_])).read()
        perf_sons_tmp = file_Process(str(_)+".txt")
        perf_sons_tmp["name"] = model_bag[str(iter_model[_])]
        model_perf_sons.append(perf_sons_tmp)
        #print(workload_perf)
    #
    return [sys_perf_counter,model_perf_sons]

def main():
    all_data = []
    all_data_trace = {}
    data_trace_count = 0#9016
    #14 models
    model_bag = {"1":"inception_v3","2":"googlenet","3":"squeezenet","4":"shufflenet","5":"vgg_11","6":"vgg_13","7":"vgg_16","8":"mobilenet_v3_small","9":"alexnet","10":"mvit","11":"mnasnet","12":"resnext50_32x4d","13":"mobilenet_v3_large","14":"mobilenet_v2","15":"segnet",'16':"efficientnet_b0"}
    key_list = range(1,17)
    model_gen_total_list = []
    #maximum on-device model number 3+1
    for _ in range(1,3):
        model_gen_total_list += list(itertools.combinations_with_replacement(key_list,_))
    model_gen_total_list = [a for a in model_gen_total_list if (15 in a or 16 in a)]
    #model_gen_total_list = model_gen_total_list[37:]
    #print(model_gen_total_list)
    #sys.exit(1)
    #model_gen_total_list = [(1,4,14)]
    for iter_model in model_gen_total_list:
        #print(len(model_gen_total_list))
        #sys.exit(1)
        #background model running, getting pid
        pid_list = []
        print("Running background workloads set:")
        print(iter_model)
        for model_id in iter_model:
            model_name = model_bag[str(model_id)]   
            #background models running
            python_cmd = ['python', 'bg_models.py', '--name', model_name, '--bs', '1']
            p = subprocess.Popen(python_cmd, stdin = subprocess.PIPE, stdout=subprocess.PIPE)
            id = p.pid
            pid_list.append(id)
            print("running {}, pid {}".format(model_name,id))
            
        print("Background workloads deployed")
        time.sleep(3)
        #record background sys info
        perf_record = {}
        
        sys_perf_counter,model_perf_sons = profiler(pid_list,iter_model)
        perf_record['global'] = [sys_perf_counter]
        perf_record['model-sons'] = model_perf_sons
        #time.sleep(10000)

        #print("background profiling results")
        #print(perf_record)

        #Run Inference Task and Profiling
        for _ in range(1,15):
            all_data_trace = {}
            print("Running inference workload:{}".format(model_bag[str(_)]))
            lat = inference(model_bag[str(_)],1)
            data_trace_count += 1
            perf_record['inference-model-info'] = 1
            gpu = GPUtil.getGPUs()[0]
            gpu_memory = round(gpu.memoryUtil,2)
            gpu_util = round(gpu.load,2)
            perf_record['inference-model-sys-info'] = {"GPU_Util":gpu_util,"GPU_Mem":gpu_memory}
            perf_record['inference-model-latency'] = lat
            perf_record['inference-model-name'] = model_bag[str(_)]
            all_data_trace[str(data_trace_count)] = perf_record

            # datapoint save
            data_obj = json.dumps(all_data_trace)
            filename = "tmpdb/"+str(data_trace_count)+"-"+str(iter_model)+"-"+str(_)+".json"
            fileObject = open(filename, 'a') 
            fileObject.write(data_obj) 
            #print("Data {} saved".format(filename))
            fileObject.close()

            print("Data point {} collected".format(str(data_trace_count)))
            time.sleep(2)
            
        #Terminate Background Model
        for _ in range(0,len(pid_list)):
            os.system('kill -9 {}'.format(pid_list[_]))
    
        time.sleep(2)
        
    #print(all_data_trace)
    #sys.exit(1)
    pass


if __name__ == '__main__':

    main()