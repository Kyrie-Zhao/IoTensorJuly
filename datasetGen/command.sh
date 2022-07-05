perf stat -p 3685 -e cpu-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-store-misses,LLC-stores,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses -a sleep 3 
perf stat -p 3686 -e cpu-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-store-misses,LLC-stores,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses -a sleep 3 
perf stat -e cpu-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-store-misses,LLC-stores,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses -a sleep 3 

"""



    16,091,525,963      cpu-cycles                                                    (33.12%)
    29,979,213,587      instructions              #    1.86  insn per cycle           (41.51%)
        85,143,385      cache-references                                              (41.63%)
        11,178,439      cache-misses              #   13.129 % of all cache refs      (41.77%)
         5,868,071      LLC-loads                                                     (41.90%)
           655,029      LLC-load-misses           #   11.16% of all LL-cache hits     (41.98%)
         7,283,111      LLC-store-misses                                              (16.78%)
        10,505,879      LLC-stores                                                    (16.65%)
       131,361,039      L1-dcache-load-misses     #    1.26% of all L1-dcache hits    (24.92%)
    10,406,066,393      L1-dcache-loads                                               (33.19%)
     5,264,480,131      L1-dcache-stores                                              (33.05%)
        76,571,733      L1-icache-load-misses                                         (33.06%)
"""