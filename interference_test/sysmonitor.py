import psutil
import time
import os
import sys
import psutil as p

pid = os.getpid()
p = psutil.Process(pid)

while(1):
    #print(p.cpu_percent(interval=1))
    print(psutil.cpu_percent(interval=0.1, percpu=True))
cpu_info_recorder = []
memory_recorder = []
for i in range(0,5,1):
    a = psutil.cpu_times_percent(interval=None, percpu=True)
    print(a)
    cpu_info_recorder.append(psutil.cpu_times_percent(interval=None, percpu=True))
    memory_recorder.append(str(psutil.virtual_memory()))
    #print(psutil.cpu_times_percent(interval=0.1, percpu=True))
    #print(str(psutil.virtual_memory()))
    time.sleep(0.5)

#print(memory_recorder)
#print(cpu_info_recorder)