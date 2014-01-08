import numpy as np
import struct as st

def load_mnist_train():
    train_data_file = 'MNIST/train-images-idx3-ubyte'
    train_labels_file = 'MNIST/train-labels-idx1-ubyte'
    data = load_images(train_data_file)
    labels = load_labels(train_labels_file)
    return data,labels

def load_mnist_test():
    train_data_file = 'MNIST/t10k-images-idx3-ubyte'
    train_labels_file = 'MNIST/t10k-labels-idx1-ubyte'
    data = load_images(train_data_file)
    labels = load_labels(train_labels_file)
    return data,labels

def load_images(fname=None):
    assert fname is not None, "No file specified"

    fid = open(fname,'r')

    magic = fid.read(4)
    numims = st.unpack('>I',fid.read(4))[0]
    cols = st.unpack('>I',fid.read(4))[0]
    rows = st.unpack('>I',fid.read(4))[0]

    rawdat = np.fromfile(fid,dtype=np.uint8)
    rawdat = rawdat.astype(np.float64)
    rawdat = rawdat.reshape(numims,rows,cols)
    rawdat = rawdat.transpose(1,2,0)
    # rescale to [0,1]
    rawdat = rawdat / np.max(rawdat)
    return rawdat

def load_labels(fname=None):
    
    fid = open(fname,'r')
    magic = fid.read(4)
    numims = st.unpack('>I',fid.read(4))[0]

    labels = np.fromfile(fid,dtype=np.uint8)
    labels = labels.astype(np.int32)

    return labels

if __name__=='__main__':
    train_data_file = 'MNIST/train-images-idx3-ubyte'
    train_labels_file = 'MNIST/train-labels-idx1-ubyte'
    data = load_images(train_data_file)
    labels = load_labels(train_labels_file)
    print "Shape of data is ",data.shape
    print "Shape of labels is ",labels.shape
