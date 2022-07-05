"""import torch
import torchvision
import time
import numpy as np
input = torch.randn(1,3,299,299).cuda()
net = torchvision.models.inception_v3(pretrained=True).cuda().eval()
time_tmp=[]
for i in range(0,100):
    time_start = time.perf_counter()
    _ = net(input)
    time_end = time.perf_counter()
    time_tmp.append(time_end-time_start)
print(np.mean(time_tmp[20:60]))
"""

import torch 
import torchvision
from emodel.mobilevit import mobilevit_s
from torchstat import stat
name = "mnasnet"
if (name=="alexnet"):
    net = torchvision.models.alexnet(pretrained=True)#.cuda().eval()
    input = torch.randn(1,3,224,224).cuda()
if (name=="vgg_11"):
    net = torchvision.models.vgg11(pretrained=True)#.cuda().eval()
    input = torch.randn(1,3,224,224).cuda()
if (name=="inception_v3"):
    input = torch.randn(1,3,299,299).cuda()
    net = torchvision.models.inception_v3(pretrained=True)
if (name=="squeezenet"):
    input = torch.randn(1,3,224,224).cuda()
    net = torchvision.models.squeezenet1_0(pretrained=True)#.cuda().eval()
if (name=="mnasnet"):
    input = torch.randn(1,3,224,224).cuda()
    net = torchvision.models.mnasnet1_0(pretrained=True)#.cuda().eval()
if (name=="mvit"):
    from emodel.mobilevit import mobilevit_s
    input = torch.randn(1,3,256,256)#.cuda()
    net = mobilevit_s()#.cuda()

stat(net,(3,224,224))