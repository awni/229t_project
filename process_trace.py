import os
import pickle

# list trace files
from os import listdir
from os.path import isfile
import re
p = re.compile('trace[0-9]*.pk', re.IGNORECASE)
tracefiles = [ f for f in listdir('.') if isfile(f) and p.match(f)!=None ]

costfile = open("cost.txt","w")
gradfile = open("grad.txt","w")
paramfile = open("param.txt","w")

for f in tracefiles:
	# process each file
	print 'processing %s...' % (f)

	tf = open(f)

	# write cost
	c = pickle.load(tf)
	for v in c: costfile.write('%f ' % (v))
	costfile.write('\n')

	# write gradient
	gl = pickle.load(tf)
	for g in gl: 
		for v in g: gradfile.write('%f ' % (v))
		gradfile.write('\n')

	# write params
	pl = pickle.load(tf)
	for p in pl:
		for v in p: paramfile.write('%f ' % (v))
		paramfile.write('\n')

costfile.close()
gradfile.close()
paramfile.close()