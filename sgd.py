
import numpy as np
import gnumpy as gp
import pickle
# gp.board_id_to_use = 1

class SGD:

    def __init__(self,model,momentum=0.9,alpha=1e-2,
                 minibatch=256):
        
        self.model = model

        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.momentum = momentum # momentum
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.velocity = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
                         for w,b in self.model.stack]

        # traces for cost, param and grad
        self.costt = []; self.paramt = []; self.gradt=[]
        # try loading previous epochs
        import os
        if os.path.isfile('trace%d.pk'%(self.model.getSeed())):
            trace = open('trace%d.pk'%(self.model.getSeed()))
            pickle.load(self.costt,trace)
            pickle.load(self.gradt,trace)
            pickle.load(self.paramt,trace)
            trace.close();

    def run(self,data,labels=None):
        """
        Runs stochastic gradient descent with model as objective.  Expects
        data in n x m matrix where n is feature dimension and m is number of
        training examples
        """
        m = data.shape[1]
        
        # momentum setup
        momIncrease = 500
        mom = 0.5

        # randomly select minibatch
        perm = np.random.permutation(range(m))

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1

            mb_data = data[:,perm[i:i+self.minibatch]]
            mb_data = gp.garray(mb_data)
                
            if labels is None:
                cost,grad = self.model.costAndGrad(mb_data)
            else:
                mb_labels = labels[perm[i:i+self.minibatch]]
                cost,grad = self.model.costAndGrad(mb_data,mb_labels)

            if self.it > momIncrease:
                mom = self.momentum

            # update velocity
            self.velocity = [[mom*vs[0]+self.alpha*g[0],mom*vs[1]+self.alpha*g[1]]
                             for vs,g in zip(self.velocity,grad)]

            # update params
            self.model.updateParams(-1.0,self.velocity)

            if self.it%10 == 0:
                print "Cost on iteration %d is %f."%(self.it,cost)
                self.costt.append(cost)
                self.gradt.append(self.model.vectorize(grad))
                self.paramt.append(self.model.paramVec())

                if self.it % 100 == 0:
                    trace = open('trace%d.pk'%(self.model.getSeed()),'w')
                    pickle.dump(self.costt,trace)
                    pickle.dump(self.gradt,trace)
                    pickle.dump(self.paramt,trace)
                    trace.close();
                    print 'Trace dumped at iteration %d' % (self.it)

            
