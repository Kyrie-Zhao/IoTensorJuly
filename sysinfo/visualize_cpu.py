import json
import numpy
import sys
import matplotlib

import matplotlib.pyplot as plt
f_cpu = open('vgg_inception_cpu_cpu.json')
data_cpu = json.load(f_cpu)
data_cpu = eval(data_cpu)
print(data_cpu)
cpu_0 = [i[0] for i in data_cpu]
cpu_1 = [i[1] for i in data_cpu]
cpu_2 = [i[2] for i in data_cpu]
cpu_3 = [i[3] for i in data_cpu]
cpu_4 = [i[4] for i in data_cpu]
cpu_5 = [i[5] for i in data_cpu]
cpu_6 = [i[6] for i in data_cpu]
cpu_7 = [i[7] for i in data_cpu]
cpu_8 = [i[8] for i in data_cpu]
cpu_9 = [i[9] for i in data_cpu]
cpu_10 = [i[10] for i in data_cpu]
cpu_11 = [i[11] for i in data_cpu]
cpu_12 = [i[12] for i in data_cpu]
cpu_13 = [i[13] for i in data_cpu]
cpu_14 = [i[14] for i in data_cpu]
cpu_15 = [i[15] for i in data_cpu]
x = [x for x in range(0,len(data_cpu))]

plt.plot(x, cpu_0, marker='x')
plt.plot(x, cpu_1, marker='x')
plt.plot(x, cpu_2, marker='x')
plt.plot(x, cpu_3, marker='x')
plt.plot(x, cpu_4, marker='x')
plt.plot(x, cpu_5, marker='x')
plt.plot(x, cpu_6, marker='x')
plt.plot(x, cpu_7, marker='x')
plt.plot(x, cpu_8, marker='x')
plt.plot(x, cpu_9, marker='x')
plt.plot(x, cpu_10, marker='x')
plt.plot(x, cpu_11, marker='x')
plt.plot(x, cpu_12, marker='x')
plt.plot(x, cpu_13, marker='x')
plt.plot(x, cpu_14, marker='x')
plt.plot(x, cpu_15, marker='x')

plt.xlim([0, len(x)+1])
plt.ylim([0, 105])
plt.xlabel('Time Series')
plt.ylabel('Utilization(%)')
plt.grid()
plt.legend(['cpu0', 'cpu1','cpu2', 'cpu3','cpu4', 'cpu5','cpu6', 'cpu7','cpu8', 'cpu9','cpu10', 'cpu11','cpu12', 'cpu13','cpu14', 'cpu15'],bbox_to_anchor=(1, 1),loc='upper left',ncol=1)
plt.savefig('vgg_inception_cpu_cpu.pdf',bbox_inches='tight',pad_inches=0.1)
#plt.show()
