import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from math import *

def loadInfo(fileName):
    with open(fileName,'r') as fid:
	opts = pickle.load(fid)
	costs = pickle.load(fid)
    return opts,costs

def smooth(costs,winSize=20):
    """
    Smooth costs using windowed average
    """
    newC = np.convolve(costs,np.ones((winSize))/float(winSize),'valid')
    return newC

def loadAll(files):
    cList = []
    for f in files:
	_,costs = loadInfo(f)
	newC = smooth(costs)
	cList.append(newC)
    return np.vstack(cList)

files = []
formatf = "../../models/ae_%s_step_%.3f_mom_%.3f_anneal_%.3f.bin"

files.append(formatf%('nesterov',1e-2,.99,1))
files.append(formatf%('adagrad',5e-3,1,1))
files.append(formatf%('adagrad3',5e-3,1,1))
files.append(formatf%('adadelta',5e-3,1,1))
allCurves = loadAll(files).T

#pretty plotting code for latex from http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

fig_width_pt = 246.0 
dth_pt = 246.0
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean + 1.0      # height in inches
fig_size =  [fig_width+0.5,fig_height]
params = {'backend': 'ps',
           'axes.labelsize': 8,
           'text.fontsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'text.usetex': True,
           'figure.figsize': fig_size}
plt.rcParams.update(params)

plt.figure(1)
plt.clf()
plt.ylim(ymax = 10)
plt.plot(allCurves[:,0],'-b',label='NAG',linewidth=0.5)
plt.plot(allCurves[:,1],'-g',label='Adagrad',linewidth=0.5)
plt.plot(allCurves[:,2],'-k',label='Adagrad3',linewidth=0.5)
plt.plot(allCurves[:,3],'-r',label='AdaDelta',linewidth=0.5)
plt.xlabel('Iteration')
plt.ylabel('Training Error')
plt.legend()
plt.savefig('mnist_ae.pdf')

