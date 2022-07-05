import os
import numpy as np
import ios
import torch
import time
import argparse
import sys
import ios.tvm_utils
from ios.ir import Graph
import json
#import ios.ir.graph as graph

def main(mode,name,optimize_type):
    expr_dir = f"outputs/interference_squeeze-1"
    
    if name=="inception_v3":
        graph = ios.models.inception_v3()
    if name=="vgg_11":
        graph = ios.models.vgg_11()
    if name=="vgg_13":
        graph = ios.models.vgg_13()
    if name=="vgg_16":
        graph = ios.models.vgg_16()
    if name=="vgg_19":
        graph = ios.models.vgg_19()
    if name=="alexnet":
        graph = ios.models.alexnet()
    if name=="squeezenet":
        graph = ios.models.squeezenet()
    if name=="nasnet_large":
        graph = ios.models.nasnet_large()
    if name=="randwire_large":
        graph = ios.models.randwire_large()

    # optimization mode
    if mode==0:
        if optimize_type == "seq":
            graph.sequential_schedule()
            seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)
            ios.draw(graph, fname=f'outputs/draw/seq_{graph.name}.png', label=f'Seq Graph, Latency = {np.mean(seq_latency):.3f}')
            print(f"seq latency = {seq_latency}")
            with open(f"{expr_dir}/{graph.name}_seq.json", "w") as f:
                json.dump(graph.export_config(), f, indent=2)
                print(f"Seq graph {name} saved")
                sys.exit(1)

        if optimize_type == "opt":
            opt_graph = ios.optimize(graph, batch_size=1)#,verbose=True)
            #sys.exit(1)
            opt_lat = ios.ios_runtime.graph_latency(optimized_graph,batch_size=1,repeat = 6)
            ios.draw(optimized_graph, fname=f'outputs/draw/optimized_{graph.name}.png', label=f'Optimized Graph, Latency = {np.mean(opt_lat):.3f}')
            with open(f"{expr_dir}/{graph.name}_opt.json", "w") as f:
                json.dump(optimized_graph.export_config(), f, indent=2)
                print(f"Optimized graph {name} saved")
                sys.exit(1)
        
        if optimize_type == "opt_code":
            import tvm
            from tvm import relay, auto_scheduler
            #import tvm.relay.testing
            from tvm.contrib import graph_runtime
            #opt_graph = ios.optimize(graph, batch_size=1)#,verbose=True)
            #sys.exit(1)
            #opt_lat = ios.ios_runtime.graph_latency(opt_graph,batch_size=1,repeat = 6)
            #ios.draw(opt_graph, fname=f'outputs/interference_squeeze-4/draw/optimized_{graph.name}.png', label=f'Optimized Graph, Latency = {np.mean(opt_lat):.3f}')
       
            """with open(f"{expr_dir}/{graph.name}_opt.json", "w") as f:
                json.dump(opt_graph.export_config(), f, indent=2)
                print(f"Optimized graph {name} saved")"""

            
            with open(f"{expr_dir}/{graph.name}_opt.json","r") as f:
                opt_graph = Graph.from_config(json.load(f))
            opt_lat = ios.ios_runtime.graph_latency(opt_graph,batch_size=1,repeat = 6)
            
            ios.draw(opt_graph, fname=f'outputs/interference_2squeeze-1/draw/optimized_{graph.name}.png', label=f'Optimized Graph, Latency = {np.mean(opt_lat):.3f}')
       
            opt_graph = ios.models.inception_v3()
            opt_graph.init_weights()
 
            print("optimized")
            network = name
            batch_size = 1
            layout = "NHWC"
            target = tvm.target.Target("cuda")
            dtype = "float32"
            log_file = f"{expr_dir}/%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
            lib_name = f"{expr_dir}/%s-%s-lib.so" % (network, layout)
            input_shape = (1,3,299,299)
            
            #opt_latency = ios.ios_runtime.graph_latency(opt_graph, batch_size=1, repeat=6)
            mod, params = ios.tvm_utils.graph2relay(opt_graph, batch_size=1)
            
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
            #print(tasks)
            #print(len(tasks))
            #sys.exit(1)
            print("Begin tuning...")
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)   

            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)#,load_model_file="xgb.pkl")
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20000,  # change this to 20000 to achieve the best performance
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )

            #tuner.tune(tune_option)
            print("Compiled...")
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                    lib = relay.build(mod, target=target, params=params)
                    
                    lib.export_library(lib_name)
            sys.exit(1)

    # inference mode
    else:
        if optimize_type == "raw":
            import torchvision
            input = torch.randn(1,3,299,299).cuda()
            if (name=="vgg_11"):
                net = torchvision.models.vgg11(pretrained=True).cuda().eval()
            if (name=="inception_v3"):
                net = torchvision.models.inception_v3(pretrained=True).cuda().eval()
            time_tmp=[]
            #for i in range(0,100):
            while(1):
                time_start = time.perf_counter()
                _ = net(input)
                time_end = time.perf_counter()
                time_tmp.append(time_end-time_start)
            print(f'Raw Inference: {(np.mean(time_tmp)*1000):.3f} ms')
        
        if optimize_type == "seq":
            with open(f"{expr_dir}/{graph.name}_seq.json","r") as f:
                seq_graph = Graph.from_config(json.load(f))
                seq_latency = ios.ios_runtime.graph_latency(seq_graph, batch_size=1, repeat=6)
                print(f' Sequential schedule: {np.mean(seq_latency):.3f} ms')

        if optimize_type == "opt":
            with open(f"{expr_dir}/{graph.name}_opt.json","r") as f:
                opt_graph = Graph.from_config(json.load(f))
                print(opt_graph.memory_access())
                print(opt_graph.kernels())
                print(opt_graph.schedule_operators())
                opt_latency = ios.ios_runtime.graph_latency(opt_graph, batch_size=1, repeat=6)
                print(f' Optimized schedule: {np.mean(opt_latency):.3f} ms')

        if optimize_type == "opt_code":
            import tvm
            from tvm import relay, auto_scheduler
            #import tvm.relay.testing
            from tvm.contrib import graph_runtime
            with open(f"{expr_dir}/{graph.name}_opt.json","r") as f:
                opt_graph = Graph.from_config(json.load(f))
            print("optimized inception")
            opt_graph.init_weights()
            network = name
            batch_size = 1
            layout = "NHWC"
            target = tvm.target.Target("cuda")
            dtype = "float32"
            log_file = f"{expr_dir}/%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
            lib_name = f"{expr_dir}/%s-%s-lib.so" % (network, layout)
            input_shape = (1,3,299,299)
            
            #opt_latency = ios.ios_runtime.graph_latency(opt_graph, batch_size=1, repeat=6)
            mod, params = ios.tvm_utils.graph2relay(opt_graph, batch_size=1)

            ctx = tvm.context(str(target), 0)
            lib: tvm.runtime.Module = tvm.runtime.load_module(lib_name)
            print("loaded...")
            gmod = graph_runtime.GraphModule(lib["default"](ctx))
        # use the graph module.
            
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            #gmod.set_input("0", data_tvm)
            time_tmp=[]
            #for i in range(0,100):
            while(1):
                time_start = time.perf_counter()
                _ = gmod.run()
                time_end = time.perf_counter()
                time_tmp.append(time_end-time_start)
            time_tmp.sort()
            print(f'Code+Parallel Inference: {(np.mean(time_tmp[:20])*1000):.3f} ms')
            #gmod.run()
            sys.exit(1)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, required=False,default = f"inception_v3", choices=["inception_v3","randwire_large","squeezenet","nasnet_large","vgg_11","vgg_13","vgg_16","vgg_19","alexnet"])
    argparser.add_argument('--optimize', type=str, required=False, choices=['raw', 'seq', 'opt', 'opt_code'],default="null")
    argparser.add_argument('--mode',type=int, required=True, choices=[0,1])
    #0: optimize 1: inference   
    args = argparser.parse_args()

    mode = args.mode
    name = args.name
    optimize_type = args.optimize


    main(mode,name,optimize_type)
