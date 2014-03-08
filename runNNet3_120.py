import sgd
import nnet
import dataLoader as dl
import numpy as np
import gnumpy as gp
import preprocess as pc

gp.board_id_to_use = 1

def run():
    print "Loading data..."
    # load training data
    trainImages,trainLabels=dl.load_mnist_train()

    imDim = trainImages.shape[0]
    inputDim = 50
    outputDim = 10
    layerSizes = [16]*3

    trainImages = trainImages.reshape(imDim**2,-1)

    pcer = pc.Preprocess()
    pcer.computePCA(trainImages)
    whitenedTrain = pcer.whiten(trainImages, inputDim)

    minibatch = 120
    print "minibatch size: %d" % (minibatch)
    epochs = 20
    stepSize = 1e-2

    nn = nnet.NNet(inputDim,outputDim,layerSizes,minibatch)
    nn.initParams()

    SGD = sgd.SGD(nn,alpha=stepSize,minibatch=minibatch)

    for e in range(epochs):
    	print "Running epoch %d"%e
    	SGD.run(whitenedTrain,trainLabels)

    SGD.dumptrace()


if __name__=='__main__':
    run()


