#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3
# echo "Bash version ${BASH_VERSION}..."
for i in {0..10..2}
do
    echo "Welcome $i times"
done


for batchsize in 8 10 16 32
do
    echo $batchsize
done
