#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3
echo "Bash version ${BASH_VERSION}..."
for i in {0..10..2}
    do
        echo "Welcome $i times"
    done


for power in $(seq 2 6)
batchsize=2**$power
do
    echo '$batchsize'
done
