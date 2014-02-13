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
formatf = "../../models/%s_step_%.3f_mom_%.3f_anneal_%.3f.bin"

files.append("../../models/sgd_step_"+str(1e-1)+"_mom_"+str(1)+".bin")
files.append("../../models/momentum_step_"+str(1e-1)+"_mom_"+str(0.9)+".bin")
files.append(formatf%('nesterov',.3,.9,1))
files.append(formatf%('adagrad',.1,1,1))
allCurves = loadAll(files).T

files2 = []
files2.append(formatf%('nesterov',.3,.9,1))
files2.append(formatf%('adagrad',.1,1,1))
files2.append(formatf%('adaccel2',.05,0.9,1))
bestCurves = loadAll(files2).T

#pretty plotting code for latex from http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

fig_width_pt = 246.0 
dth_pt = 246.0
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
           'axes.labelsize': 10,
           'text.fontsize': 10,
           'legend.fontsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8,
           'text.usetex': True,
           'figure.figsize': fig_size}
plt.rcParams.update(params)

plt.figure(1)
plt.clf()
plt.ylim(ymax = 0.5)
plt.plot(allCurves[:,0],'-b',label='SGD',linewidth=0.5)
plt.plot(allCurves[:,1],'-g',label='CM',linewidth=0.5)
plt.plot(allCurves[:,2],'-k',label='NAG',linewidth=0.5)
plt.plot(allCurves[:,3],'-r',label='AdaGrad',linewidth=0.5)
plt.legend()
plt.savefig('allcurves.pdf')

plt.figure(1)
plt.clf()
plt.ylim(ymax = 0.01)
plt.xlim(xmin = 1000, xmax = 7000)
plt.plot(bestCurves[:,0],'-b',label='NAG',linewidth=0.5)
plt.plot(bestCurves[:,1],'-g',label='AdaGrad',linewidth=0.5)
plt.plot(bestCurves[:,2],'-k',label='Accel Ada',linewidth=0.5)
plt.legend()
plt.savefig('bestcurves.pdf')
