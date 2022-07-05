import psutil
import time
import os
import sys
import psutil as p
import json
pid = os.getpid()
p = psutil.Process(pid)
cpu_info_recorder = []
cpu_info_recorder_whole = []
memory_recorder = []
try:
    while(1):
        #print(p.cpu_percent(interval=1))
        #print(psutil.cpu_percent(interval=0.1, percpu=False))
        #print(psutil.virtual_memory())
        cpu_info_recorder_whole.append(psutil.cpu_percent(interval=0.1, percpu=False))
        cpu_info_recorder.append(psutil.cpu_percent(interval=0.1, percpu=True))
        memory_recorder.append(str(psutil.virtual_memory()))
except KeyboardInterrupt:
    print("download sys info")
    cpu = json.dumps(cpu_info_recorder)
    cpu_whole = json.dumps(cpu_info_recorder)
    memory = json.dumps(memory_recorder)
    with open() as :
        json.dump(cpu_info_recorder_whole
    with open('sysinfo/vgg_inception_cpu_cpu.json', 'w') as outfile:
        json.dump(cpu, outfile)
    with open('sysinfo/vgg_inception_cpu_memory.json', 'w') as outfile:
        json.dump(memory, outfile)
    sys.exit(1)
