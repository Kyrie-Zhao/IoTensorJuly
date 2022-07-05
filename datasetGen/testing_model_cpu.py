import torchvision
import time
import os
import numpy as np
import torch
import time
import argparse
import sys
import json
import sys
from torchstat import stat

def inference(name,batch_size):
    if (name=="inception_v3"):
        net = torchvision.models.inception_v3(pretrained=True).eval().cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,299,299).cpu()#.cuda()
    if (name=="googlenet"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.googlenet(pretrained=True).cpu()#.cuda().eval()
    if (name=="squeezenet"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        net.cpu()#.cuda().eval()
    if (name=="shufflenet"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.shufflenet_v2_x1_0(pretrained=True).cpu()#.cuda().eval()
    if (name=="vgg_11"):
        net = torchvision.models.vgg11(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
    if (name=="vgg_13"):
        net = torchvision.models.vgg13(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
    if (name=="vgg_16"):
        net = torchvision.models.vgg16(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
    if (name=="mobilenet_v3_small"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.mobilenet_v3_small().cpu()#.cuda().eval()
    if (name=="alexnet"):
        net = torchvision.models.alexnet(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
    if (name=="mvit"):
        from mobilevit import mobilevit_s
        input = torch.randn(batch_size,3,256,256).cpu()#.cuda()
        net = mobilevit_s().cpu()#.cuda()
    if (name=="mnasnet"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.mnasnet1_0(pretrained=True).cpu()#.cuda().eval()
    if (name=="resnext50_32x4d"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.resnext50_32x4d(pretrained=True).cpu()#.cuda().eval()
    if (name=="mobilenet_v3_large"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.mobilenet_v3_large().cpu()#.cuda().eval()
    if (name=="mobilenet_v2"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torchvision.models.mobilenet_v2().cpu()#.cuda().eval()
        
    


    time_tmp=[]
    pid = os.getpid()
    try:
        for _ in range(0,100):
            time_start = time.perf_counter()
            _ = net(input)
            time_end = time.perf_counter()
            time_tmp.append(round((time_end-time_start)*1000,3))
            #time.sleep(0.5)
            
        #print(f'Raw Inference: {(np.mean(time_tmp)):.3f} ms')
        #print("return results")
        return(np.mean(time_tmp[20:-20]))
    except KeyboardInterrupt:
        print("Background Model Done")
        print(f'Raw Inference: {(np.mean(time_tmp)):.3f} ms')
        """with open('sysinfo/vgg_inception_cpu_cpu.json', 'w') as outfile:
            json.dump(cpu, outfile)
        with open('sysinfo/vgg_inception_cpu_memory.json', 'w') as outfile:
            json.dump(memory, outfile)"""
        sys.exit(1)



