import torchvision
import time
import os
import numpy as np
import torch
import time
import argparse
import sys
#import affinity
import psutil
import json

def ig_module(name,batch_size):
    if (name=="mnasnet"):
        input = torch.randn(1,3,224,224).cpu()#.cuda()
        net = torchvision.models.mnasnet1_0(pretrained=True).cpu()#.cuda().eval()
    if (name=="alexnet"):
        net = torchvision.models.alexnet(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(1,3,224,224).cpu()#.cuda()
    if (name=="vgg_11"):
        net = torchvision.models.vgg11(pretrained=True).cpu()#.cuda().eval()
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
    if (name=="inception_v3"):
        net = torchvision.models.inception_v3(pretrained=True).cpu().eval()#.cuda().eval()
        input = torch.randn(batch_size,3,299,299).cpu()#.cuda()
    if (name=="squeezenet"):
        input = torch.randn(batch_size,3,224,224).cpu()#.cuda()
        net = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        net.cpu()#.cuda().eval()
    if (name=="mvit"):
        from mobilevit import mobilevit_s
        input = torch.randn(1,3,256,256).cpu()#.cuda()
        net = mobilevit_s().cpu()#.cuda()
    
    while(1):
        _ = net(input)
        #a = 1
        #time.sleep(0.5)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, required=False,default = f"inception_v3", choices=["inception_v3","randwire_large","squeezenet","nasnet_large","vgg_11","vgg_13","vgg_16","vgg_19","alexnet","mvit","mnasnet"])
    argparser.add_argument('--bs',type=int, required=True, choices=[1,2,4,8,16,32])
    #0: optimize 1: inference   
    args = argparser.parse_args()

    name = args.name
    batch_size = args.bs
    ig_module(name,batch_size)