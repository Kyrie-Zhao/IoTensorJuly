#["inception_v3","randwire_large","squeezenet","nasnet_large","vgg_11",
#"vgg_13","vgg_16","vgg_19","alexnet"]

task_1_name="alexnet"
task_1_optimize="raw"
task_1_mode=1

task_2_name="squeezenet"
task_2_optimize="raw"
task_2_mode=1

task_3_name="inception_v3"
task_3_optimize="opt_code"
task_3_mode=1

#one task
python main_interferenceTest.py --name $task_1_name --optimize $task_1_optimize --mode $task_1_mode

#two tasks
#python main_interferenceTest.py --name $task_1_name --optimize $task_1_optimize --mode $task_1_mode & python main_interferenceTest.py --name $task_2_name --optimize $task_2_optimize --mode $task_2_mode

#Three tasks
#python main_interferenceTest.py --name $task_1_name --optimize $task_1_optimize --mode $task_1_mode & python main_interferenceTest.py --name $task_2_name --optimize $task_2_optimize --mode $task_2_mode & python main_interferenceTest.py --name $task_3_name --optimize $task_3_optimize --mode $task_3_mode