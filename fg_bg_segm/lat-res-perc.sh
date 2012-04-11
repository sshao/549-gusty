#!/bin/bash

#script looks at latency as function of resolution and percentage of blobs taken
#keeping ratio of dimensions the same

mkdir -p lat-res-perc
for (( w = 64; w <= 640; w += 64 ))
do
    ((h = w * 3 / 4));
    mkdir -p lat-res-perc/$w-$h
    for (( p = 5; p <= 25; p+= 5))
    do
        #echo "width = $w height = $h percentage = $p" | awk '$1=="width"{print $3,$6,$9}' >> lat-res-perc/$w-$h/$p.txt
        #echo ./blobdetect $w $h $p 5 50 16 50 & sleep 15; echo $!; kill $! | awk '$1=="total"{print $4,$5,$6,$7}' >> lat-res-perc/$w-$h/$p.txt
        #./blobdetect $w $h $p 5 50 16 50 | awk '$1=="total"{print $4,$5,$6,$7}' >> lat-res-perc/$w-$h/$p.txt & sleep 15; kill $!
        # BELOW LINE IS GOOD
        ./blobdetect $w $h $p 5 50 16 50 | awk '$1=="total"{print $4,$5,$6,$7}' >> lat-res-perc/$w-$h/$p.txt & sleep 15; pkill blobdetect
        #awk '{sum=sum+$1+$2+$3}END{print sum/NR}' lat-res-perc/$w-$h/$p.txt >> lat-res-perc/$w-$h/avg.txt
        awk '{sum+=$3}END{print sum/NR}' lat-res-perc/$w-$h/$p.txt >> lat-res-perc/$w-$h/avg.txt
    done 
    awk '{sum+=$1}END{print sum/NR}' lat-res-perc/$w-$h/avg.txt >> lat-res-perc/$w-$h-avg.txt
done
