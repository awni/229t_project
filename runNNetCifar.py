import optparse
import numpy as np
import gnumpy as gp
import cPickle as pickle

import sgd
import nnet
import dataLoaderCifar as dl
import preprocess as pp

gp.board_id_to_use = 0

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("--minibatch",dest="minibatch",type="int",default=256)
    parser.add_option("--layers",dest="layers",type="string",
	    default="100,100",help="layer1size,layer2size,...,layernsize")
    parser.add_option("--noPca",dest="pca",action="store_false",default=True)
    parser.add_option("--inputDim",dest="inputDim",type="int",default=300)
    parser.add_option("--optimizer",dest="optimizer",type="string",
	    default="momentum")
    parser.add_option("--momentum",dest="momentum",type="float",
	    default=0.9)
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)
    parser.add_option("--outFile",dest="outFile",type="string",
	    default="models_cifar/test.bin")
    parser.add_option("--anneal",dest="anneal",type="float",default=1,
	    help="factor to divide learning rate by at each epoch")

    (opts,args)=parser.parse_args(args)
    opts.layers = [int(l) for l in opts.layers.split(',')]

    print "Loading data..."
    # load training data
    trainImages,trainLabels=dl.load_train()

    outputDim = 10

    print "PCA the data to %d..."%opts.inputDim
    if opts.pca:
	pca = pp.Preprocess()
	pca.computePCA(trainImages)
	trainImages = pca.whiten(trainImages,numComponents=opts.inputDim)

    assert trainImages.shape[0] == opts.inputDim,"Dimension mismatch."

    nn = nnet.NNet(opts.inputDim,outputDim,opts.layers,opts.minibatch)
    nn.initParams()

    SGD = sgd.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
	    optimizer=opts.optimizer,momentum=opts.momentum)

    for e in range(opts.epochs):
    	print "Running epoch %d"%e
    	SGD.run(trainImages,trainLabels)
	SGD.alpha = SGD.alpha/opts.anneal

    with open(opts.outFile,'w') as fid:
	pickle.dump(opts,fid)
	pickle.dump(SGD.costt,fid)


if __name__=='__main__':
    run()


