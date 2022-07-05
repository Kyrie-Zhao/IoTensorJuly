for modelist
        python_cmd = "python fs -ls %s"%(hive_tb_path)
        subprocess.Popen('脚本/shell', shell=True)
        p_1 = subprocess.Popen(hadoop_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #ret = p.wait() 
        id = p.pid

        process modelist 
        get process id

        wait 2 S
        collect system resource (l1 cache )
        collect new model (arch info)
        
        call input model 
        collect model latency
        save recordes json(background model:[1,2,3],inputmodel:1, inputmodelinfo: (flops:1,mac:1,branch:1,)system resource: ["l1":1,"l2",....],)
        terminate id
    pass


[hardware info + arch info] => latency degradation
{
  "background": {
      "background_model_id":[],
      ""

  }
  "input": {
    "input_model_id": x,
    "flops": "0",
    "memory_RW": "0",
    "layer_number": x,
    "branch_number":
  },
  "blocks": [


{"background_id":}


loop:
background running (get pid x,hardware counter collector)
test model (latency degradation collect)
stop pid x

loop:
test model + background (hardware counter, test model latency collect)

model info + hardware counter?

14model 20000 kernel [] input hw ]

nnmeter => model -> latency 
model + resource contention(hardware scheduling) => latency degrad

regressor:


target model (unseen)


[background(target?+background) (hardware) + target model info (flops, macs, conv, memory r/w)] => target latency degradation <> workload intensy
                                                                                    => 5 

y: latency <>
x: workload <>

0  inf
*1 inf

5 档次



rmse<>

[modela modelb modelc] => hardware info
hardware info + model info => degradation
modela + background (b c)
