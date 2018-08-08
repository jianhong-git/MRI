#!/bin/bash
#!/home/jhchen/anaconda3/bin/python3
# echo "Bash version ${BASH_VERSION}..."

for batchsize in 8 10 16 32 #{0..10..2}
let 'c=2**$batchsize'
do
    echo "$batchsize"  #"Welcome $i times" #$batchsize
    echo $c
done
