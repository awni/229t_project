import numpy as np
import struct as st
import cPickle

BASE="cifar-10-batches-py/"

def load_train(num_files=5):
    """
    Loads num_files (max=5) of CIFAR-10 training files.
    """
    if num_files > 5:
	num_files = 5

    train_file = BASE+'data_batch_%d'
    data = []; labels = []
    for i in range(1,num_files+1):
	d,l = unpickle(train_file%i)
	d = d.astype(np.float32)
	data.append(grayscale(d))
	labels += l
    data = np.vstack(data)
    labels = np.array(labels)
    return data.T,labels

def load_test():
    """
    CIFAR-10 Test file.
    """
    test_file = BASE+'test_batch'
    data,labels = unpickle(test_file)
    return grayscale(data).astype(np.float32).T,np.array(labels)

def grayscale(data):
    """
    Gray scale data (samples x [r g b]).
    """
    r = 0.2126; g = 0.7152; b = 0.0722
    data = r*data[:,:1024] + g*data[:,1024:2048] + b*data[:,2048:]
    return data

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data'],dict['labels']

if __name__=='__main__':
    data,labels = load_train (num_files=2)
    print "Shape of training data is ",data.shape
    print "Shape of training labels is ",labels.shape
    data,labels = load_test ()
    print "Shape of test data is ",data.shape
    print "Shape of test labels is ",labels.shape

