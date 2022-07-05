#["inception_v3","randwire_large","squeezenet","nasnet_large","vgg_11",
#"vgg_13","vgg_16","vgg_19","alexnet"]
#"[{'(6,)'}, {'(4, 7, 8)'}, {'(2, 2, 8)'}, {'(1, 11, 14)'}, {'(2, 9, 9)'}, {'(6, 9, 10)'}]"
#model_bag = {"1":"inception_v3","2":"googlenet","3":"squeezenet","4":"shufflenet","5":"vgg_11","6":"vgg_13","7":"vgg_16","8":"mobilenet_v3_small","9":"alexnet","10":"mvit","11":"mnasnet","12":"resnext50_32x4d","13":"mobilenet_v3_large","14":"mobilenet_v2","15":"segnet",'16':"efficientnet_b0"}
#(8),(4,14),(8, 8, 8) 

interference_1_name="alexnet"
interference_1_bs=1

interference_2_name="mobilenet_v3_small"
interference_2_bs=1

interference_3_name="mobilenet_v3_small"
interference_3_bs=1

#python sysmonitor.py
#one interference task
python interference.py --bs $interference_1_bs --name $interference_1_name 
#python interference_cpu.py --bs $interference_1_bs --name $interference_1_name 

#two interference tasks
#python interference.py --name $interference_1_name --bs $interference_1_bs & python interference.py --name $interference_2_name --bs $interference_2_bs
#python interference_cpu.py --name $interference_1_name --bs $interference_1_bs & python interference_cpu.py --name $interference_2_name --bs $interference_2_bs

#three interference tasks 
#python interference.py --bs $interference_1_bs --name $interference_1_name & python interference.py --bs $interference_2_bs --name $interference_2_name & python interference.py --bs $interference_3_bs --name $interference_3_name 
#python interference_cpu.py --name $interference_1_name --bs $interference_1_bs & python interference_cpu.py --name $interference_1_name --bs $interference_2_bs & python interference_cpu.py --name $interference_2_name --bs $interference_1_bs 

#four interference tasks 
#python interference_cpu.py --name $interference_1_name --bs $interference_1_bs & python interference_cpu.py --name $interference_1_name --bs $interference_1_bs & python interference_cpu.py --name $interference_1_name --bs $interference_1_bs & python interference_cpu.py --name $interference_1_name --bs $interference_1_bs 