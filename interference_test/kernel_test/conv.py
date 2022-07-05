import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import sys
from torchstat import stat

import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      #self.conv2 = nn.Conv2d(32, 64, 3, 1)
      #self.dropout1 = nn.Dropout2d(0.25)
      #self.dropout2 = nn.Dropout2d(0.5)
      #self.fc1 = nn.Linear(9216, 128)
      #self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      #x = F.relu(x)
      #x = self.conv2(x)
      #x = F.relu(x)

      # Run max pooling over x
      #x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      #x = self.dropout1(x)
      # Flatten x with start_dim=1
      #x = torch.flatten(x, 1)
      # Pass data through fc1
      #x = self.fc1(x)
      #x = F.relu(x)
      #x = self.dropout2(x)
      #x = self.fc2(x)

      # Apply softmax to x
      #output = F.log_softmax(x, dim=1)
      return x#output


random_data = torch.rand((16, 1, 28, 28)).cuda()
device = torch.device('cuda')
net = Net().to(device)

#random_data = torch.rand((64, 1, 28, 28)).cpu()
#con = Net().cpu()
time_tmp=[]
#for i in range(0,100):
#print(torch.cuda.memory_stats(device=None))
#stat(net,(1,28,28))
#sys.exit(1)
while(1):
  time_start = time.perf_counter()
  result = net(random_data)
          
  time_end = time.perf_counter()
  time_tmp.append(time_end-time_start)
sys.exit(1) 
with torch.autograd.profiler.emit_nvtx():
  try:
      #for i in range(0,1):
      while(1):
          time_start = time.perf_counter()
          result = net(random_data)
          
          time_end = time.perf_counter()
          time_tmp.append(time_end-time_start)
          #print(f'Raw Inference: {(np.mean(time_tmp)*1000):.3f} ms')
  except KeyboardInterrupt:
      print("Background Model Done")
      print(f'Raw Inference: {(np.mean(time_tmp)*1000):.3f} ms')
      print(f'Sample Number: {(len(time_tmp))}')
      print(f'Raw Inference STD: {(np.std(time_tmp)*1000):.3f} ms')
  print("Background CONV Done")
  #print(torch.cuda.memory_stats(device=None))
  print(f'Raw Inference: {(np.mean(time_tmp)*1000):.3f} ms')
  print(f'Sample Number: {(len(time_tmp))}')
  print(f'Raw Inference STD: {(np.std(time_tmp)*1000):.3f} ms')
#print (result)