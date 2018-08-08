#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3

for batchsize in {2**'seq 1'}; #'seq 10';{1..10};$(seq 2 6)
do
    for epochs in 1000;
    do
        echo "CUDA_VISIBLE_DEVICES=2,3 /bin/python3 fcarg.py --batchsize=${batchsize} --epochs=${epochs} "
        CUDA_VISIBLE_DEVICES=2,3 /bin/python3 fcarg.py --batchsize=${batchsize} --epochs=${epochs}
    done
done
