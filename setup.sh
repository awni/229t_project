#!/bin/bash

# Download and uncompress MNIST
mkdir -p MNIST
cd MNIST

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

for x in `ls .`
do
    gunzip $x
done

# Download and uncompress CIFAR-10
curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm -f cifar-10-python.tar.gz
