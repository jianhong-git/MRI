#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3

for power in $(seq 2 6):
    batchsize=$((2**power))
    do
    echo $(batchsize)
    done
