#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3
for fc1 in {100..600..100}
do
    for fc2 in {20..120..20}
    do

        for batchsize in 1 2 4 5 8 10 16 20 32 50 64
        do
            for epochs in 500 1000
            do
                echo "CUDA_VISIBLE_DEVICES=1,3 /bin/python3 fcarg.py --fc1=${fc1} --fc2=${fc2} --batchsize=${batchsize} --epochs=${epochs} "
                CUDA_VISIBLE_DEVICES=1,3 /bin/python3 fcarg.py --fc1=${fc1} --fc2=${fc2} --batchsize=${batchsize} --epochs=${epochs}
            done
        done
    done
done
