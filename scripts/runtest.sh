#!/bin/bash -f
for threads in {16,8,4,2,1}
    do
    export OMP_NUM_THREADS=$threads
    echo "Threads ${threads}"
    export OMP_PROC_BIND=TRUE
    numactl --cpunodebind=0 ./$1/panSim --d_peak_offset -5 --d_offset 15 -w 10 --trunc 0.7 --infectiousnessMultiplier 1.03,1.79,2.3,2.7,3.8,4.8,3.6 --diseaseProgressionScaling 0.9,1.0,1.2,0.75,0.5,0.45,0.8  --diseaseProgressionDeathScaling 1.0,1.03,1.2,0.6,0.4,0.6,0.6 --immunizationOrder 1,2,3,4,5,6,0,0,7,8 --vaccinationGroupLevel 0.9,0.85,0.9,0.82,0.8,0.75,0.8,0.67,0.4,0.2 -r --closures inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json --testingProbabilities 0.00007,0.01,0.0005,0.0005,0.005,0.05 --boosterStart 315 --variantSimilarity 0,0,0,1,1,1 --variantSimilarMultiplier 0.0,0.0,0.0,0.0,0.2,0.41 -N 80000 -n 1795002>> run_threadscale10w_$1.txt
done
