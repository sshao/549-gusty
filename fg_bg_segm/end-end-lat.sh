#!/bin/bash

mkdir -p end-end-lat
for (( w = 1; w <= 10; w += 1 ))
do
    mkdir -p end-end-lat/run-$w
    ./blobdetect | awk '$1=="sending"{print $4,$5,$6,$7,$8}' >> end-end-lat/run-$w/run.txt & sleep 60; pkill blobdetect; sleep 5
    # YO KEEYOUNG
    # idk why but that sleep 5 at the end fixed things
    # runs just fine now, no NaN
    awk '{sum+=$4}END{print sum/NR}' end-end-lat/run-$w/run.txt >> end-end-lat/run-$w/avg.txt
done

