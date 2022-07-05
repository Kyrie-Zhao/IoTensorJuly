import json
import numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches
f_mem = open('vgg_inception_cpu_memory.json')

data_mem = json.load(f_mem)


data_mem = eval(data_mem)
x = [x for x in range(0,len(data_mem))]

percent_r = []
cached_r = []
shared_r = []
for _ in x:
    r = re.search('percent=(.*), u', data_mem[_])
    percent = float(r.group(1))
    percent_r.append(percent)
    r = re.search('cached=(.*), sh', data_mem[_])
    cached = int(r.group(1))/(1024*1024)
    cached_r.append(cached)
    r = re.search('shared=(.*), sl', data_mem[_])
    shared = int(r.group(1))/(1024*1024)
    shared_r.append(shared)
print(percent_r)
print(cached_r)
print(shared_r)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, percent_r, marker='x',label='memory percentage',color='g')

ax2 = ax.twinx()
ax2.plot(x, cached_r, marker='x',label='cached memory')
ax2.plot(x, shared_r, marker='x',label='shared memory')
per = mpatches.Patch(color='green', label='memory percentage')
ax.legend(handles=[per],bbox_to_anchor=(1, 1),loc='upper left')
ax.grid()
ax.set_xlabel('Time Series')
ax.set_ylabel('Memory Percentage(%)')
ax2.set_ylabel('MBytes')

ax.set_ylim([0, 105])
ax2.set_ylim([0, 1.05*max(cached_r)])
ax2.legend(bbox_to_anchor=(1, 0),loc='upper left')
plt.savefig('vgg_inception_cpu_memory.pdf',bbox_inches='tight',pad_inches=0.1)
