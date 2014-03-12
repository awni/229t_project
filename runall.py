import os

optimizers = ['adagrad','nesterov'] #,'momentum','adagrad'] #['sgd','momentum','nesterov','adagrad']
#optimizers = ['adagrad3','adagrad','adaccel2','nesterov']
epochs = 60
layers="200,200,200,200,200"
steps = [1e-1,1e-2,1e-3]
momentums = [0.90,0.99] #[0.9,0.8,0.7,0.6]
anneals = [1.0]

commForm = "python runNNet.py --layers %s --optimizer %s --step %f --epochs %d \
	    --momentum %d --anneal %f --outFile %s"
outForm = "models/%s_step_%.3f_mom_%.3f_anneal_%.3f.bin"

for opt in optimizers:
    for step in steps:
	for anneal in anneals:
	    if opt is 'nesterov' or opt is 'momentum' or opt is 'adaccel2':
		moms = momentums
	    else:
		moms = [1] # dummy
	    for mom in moms:
		outfile = outForm%(opt,step,mom,anneal)
		command = commForm%(layers,opt,step,epochs,mom,anneal,outfile)
		print "running command : - %s"%command
		os.system(command) # run NNET
