nvprof -f -o net.sql python conv.py 
python -m pyprof.parse net.sql > net.dict
#python -m pyprof.prof --csv -c tid,kernel,op,params,sil,stream,grid,block,bytes,flops net.dict
python -m pyprof.prof --csv -c op,sil,stream,grid,block,bytes,flops net.dict
#python conv.py & python conv.py

#nvprof --print-gpu-trace python conv.py