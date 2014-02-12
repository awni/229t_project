import numpy as np
import gnumpy as gp
import pickle

# gp.board_id_to_use = 1

class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=256,
	         optimizer='momentum',momentum=0.9):
        self.model = model

        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.momentum = momentum # momentum
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
	self.optimizer = optimizer
	if self.optimizer == 'momentum':
	    print "Using momentum.."
	    self.velocity = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	elif self.optimizer == 'nesterov':
	    print "Using nesterov.."
	    self.velocity = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	elif self.optimizer == 'adagrad':
	    print "Using adagrad.."
	    self.gradt = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	elif self.optimizer == 'adaccel':
	    print "Using adaccel.."
	    self.gradt = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	    self.sqgradt = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	elif self.optimizer == 'adaccel2':
	    print "Using adaccel2.."
	    self.gradt = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]
	    self.velocity = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			      for w,b in self.model.stack]

	elif self.optimizer == 'sgd':
	    print "Using sgd.."
	else:
	    raise ValueError("Invalid optimizer")

	self.costt = []

    def run(self,data,labels=None):
        """
        Runs stochastic gradient descent with model as objective.  Expects
        data in n x m matrix where n is feature dimension and m is number of
        training examples
        """
        m = data.shape[1]
        
        # momentum setup
        momIncrease = 10
        mom = 0.5

        # randomly select minibatch
        perm = np.random.permutation(range(m))

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1

            mb_data = data[:,perm[i:i+self.minibatch]]
            mb_data = gp.garray(mb_data)

	    if self.optimizer == 'nesterov' or self.optimizer == 'adaccel2':
		# w = w+mom*velocity (evaluate gradient at future point)
		self.model.updateParams(mom,self.velocity)
                
            if labels is None:
                cost,grad = self.model.costAndGrad(mb_data)
            else:
                mb_labels = labels[perm[i:i+self.minibatch]]
                cost,grad = self.model.costAndGrad(mb_data,mb_labels)

	    if self.optimizer == 'momentum':
		if self.it > momIncrease:
		    mom = self.momentum
		# velocity = mom*velocity + eta*grad
		self.velocity = [[mom*vs[0]+self.alpha*g[0],mom*vs[1]+self.alpha*g[1]]
				  for vs,g in zip(self.velocity,grad)]
		update = self.velocity
		scale = -1.0

	    elif self.optimizer == 'adagrad':
		# trace = trace+grad.^2
		self.gradt = [[gt[0]+g[0]*g[0],gt[1]+g[1]*g[1]] 
			for gt,g in zip(self.gradt,grad)]
		# update = grad.*trace.^(-1/2)
		update =  [[g[0]*(1./gp.sqrt(gt[0])),g[1]*(1./gp.sqrt(gt[1]))]
			for gt,g in zip(self.gradt,grad)]
		scale = -self.alpha

	    elif self.optimizer == 'nesterov':
		# velocity = mom*velocity - alpha*grad
		self.velocity = [[mom*vs[0]-self.alpha*g[0],mom*vs[1]-self.alpha*g[1]]
				  for vs,g in zip(self.velocity,grad)]
		update = self.velocity
		scale = 1.0

	    elif self.optimizer == 'adaccel':
		# trace = trace+grad
		self.gradt = [[gt[0]+g[0],gt[1]+g[1]] 
			for gt,g in zip(self.gradt,grad)]
		# sqtrace = sqtrace+grad.^2
		self.sqgradt = [[gt[0]+g[0]*g[0],gt[1]+g[1]*g[1]] 
			for gt,g in zip(self.sqgradt,grad)]

		# update = grad.*trace.^(-1/2)
		update =  [[g[0]*(1./gp.sqrt(sqgt[0]-gt[0])),g[1]*(1./gp.sqrt(sqgt[1]-gt[1]))]
			for gt,sqgt,g in zip(self.gradt,self.sqgradt,grad)]
		scale = -self.alpha

	    elif self.optimizer == 'adaccel2':
		# velocity = mom*velocity - alpha*grad
		self.velocity = [[mom*vs[0]-self.alpha*g[0],mom*vs[1]-self.alpha*g[1]]
				  for vs,g in zip(self.velocity,grad)]
		# trace = trace+grad.^2
		self.gradt = [[gt[0]+g[0]*g[0],gt[1]+g[1]*g[1]] 
			for gt,g in zip(self.gradt,grad)]

		# update = velocity.*trace.^(-1/2)
		update =  [[v[0]*(1./gp.sqrt(gt[0])),v[1]*(1./gp.sqrt(gt[1]))]
			for gt,v in zip(self.gradt,self.velocity)]
		scale = 1.0

	    elif self.optimizer == 'sgd':
		update = grad
		scale = -self.alpha

	    # update params
	    self.model.updateParams(scale,update)

	    self.costt.append(cost)
            if self.it%10 == 0:
                print "Cost on iteration %d is %f."%(self.it,cost)
            
