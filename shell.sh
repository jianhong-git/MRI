#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3

for batchsize in 4 8 #10 16 32
do
    for epochs in 500 1000
    do
        echo "CUDA_VISIBLE_DEVICES=2,3 /bin/python3 fcarg.py --batchsize=${batchsize} --epochs=${epochs} "
        CUDA_VISIBLE_DEVICES=2,3 /bin/python3 fcarg.py --batchsize=${batchsize} --epochs=${epochs}
    done
done
