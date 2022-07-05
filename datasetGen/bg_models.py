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

def ig_module(name,batch_size):
    if (name=="inception_v3"):
        net = torchvision.models.inception_v3(pretrained=True).cuda().eval()
        input = torch.randn(batch_size,3,299,299).cuda()
    if (name=="googlenet"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.googlenet(pretrained=True).cuda().eval()
    if (name=="squeezenet"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        net.cuda().eval()
    if (name=="shufflenet"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.shufflenet_v2_x1_0(pretrained=True).cuda().eval()
    if (name=="vgg_11"):
        net = torchvision.models.vgg11(pretrained=True).cuda().eval()
        input = torch.randn(batch_size,3,224,224).cuda()
    if (name=="vgg_13"):
        net = torchvision.models.vgg13(pretrained=True).cuda().eval()
        input = torch.randn(batch_size,3,224,224).cuda()
    if (name=="vgg_16"):
        net = torchvision.models.vgg16(pretrained=True).cuda().eval()
        input = torch.randn(batch_size,3,224,224).cuda()
    if (name=="mobilenet_v3_small"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.mobilenet_v3_small().cuda().eval()
    if (name=="alexnet"):
        net = torchvision.models.alexnet(pretrained=True).cuda().eval()
        input = torch.randn(batch_size,3,224,224).cuda()
    if (name=="mvit"):
        from emodels.mobilevit import mobilevit_s
        input = torch.randn(batch_size,3,256,256).cuda()
        net = mobilevit_s().cuda()
    if (name=="mnasnet"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.mnasnet1_0(pretrained=True).cuda().eval()
    if (name=="resnext50_32x4d"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.resnext50_32x4d(pretrained=True).cuda().eval()
    if (name=="mobilenet_v3_large"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.mobilenet_v3_large().cuda().eval()
    if (name=="mobilenet_v2"):
        input = torch.randn(batch_size,3,224,224).cuda()
        net = torchvision.models.mobilenet_v2().cuda().eval()
    if (name=="segnet"):
        from emodels.SegNet import SegNet
        input = torch.randn(batch_size,3,800,600).cuda()
        net = SegNet().cuda()
    if (name=="efficientnet_b0"):
        from emodels.efficientnet import EfficientNet
        input = torch.randn(batch_size,3,224,224).cuda()
        net = EfficientNet('efficientnet_b0').cuda()
    
    time_tmp=[]
    pid = os.getpid()
    #print("fuking")
    try:
        while(1):
            time_start = time.perf_counter()
            #for _ in range(0,3):
            _ = net(input)
            time_end = time.perf_counter()
            time_tmp.append(round((time_end-time_start)*1000,3))
            #time.sleep(0.5)
        print(f'Raw Inference: {(np.mean(time_tmp)):.3f} ms')
    except KeyboardInterrupt:
        print("Background Model Done")
        print(f'Raw Inference: {(np.mean(time_tmp)):.3f} ms')
        """with open('sysinfo/vgg_inception_cpu_cpu.json', 'w') as outfile:
            json.dump(cpu, outfile)
        with open('sysinfo/vgg_inception_cpu_memory.json', 'w') as outfile:
            json.dump(memory, outfile)"""
        sys.exit(1)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, required=False,default = f"inception_v3")
    argparser.add_argument('--bs',type=int, required=True, choices=[1,2,4,8,16,32])
    #0: optimize 1: inference   
    args = argparser.parse_args()

    name = args.name
    batch_size = args.bs
    ig_module(name,batch_size)