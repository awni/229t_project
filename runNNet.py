import sgd
import nnet
import dataLoader as dl
import numpy as np

def run():
    print "Loading data..."
    # load training data
    trainImages,trainLabels=dl.load_mnist_train()

    imDim = trainImages.shape[0]
    inputDim = imDim**2
    outputDim = 10
    layerSizes = [1024]*2

    trainImages = trainImages.reshape(inputDim,-1)

    minibatch = 256
    epochs = 3
    stepSize = 1e-2

    nn = nnet.NNet(inputDim,outputDim,layerSizes,minibatch)
    nn.initParams()

    SGD = sgd.SGD(nn,alpha=stepSize,minibatch=minibatch)

    for e in range(epochs):
	print "Running epoch %d"%e
	SGD.run(trainImages,trainLabels)


if __name__=='__main__':
    run()


