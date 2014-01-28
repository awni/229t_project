
import numpy as np
import numpy.linalg as npl


class Preprocess:
    def __init__(self):
        self.U = None # eigenvectors
        self.V = None # eigenvalues
        self.meanPatch = None
        self.epsilon = 1e-4

    def computePCA(self,data):
        """
        Computes the mean of the data for centering. As well
        as the eigenvalue decomposition of the sample
        covariance matrix of the centered data.
        """
        self.meanPatch = np.mean(data,axis=1)
        self.meanPatch = self.meanPatch.reshape(-1,1)

        tmpData = data-self.meanPatch

        sigma = np.cov(data)
        self.V,self.U = npl.eigh(sigma)
        self.V = np.flipud(self.V)
        self.U = np.fliplr(self.U)

    def whiten(self,data,numComponents=-1):
        """
        Whitens the data by left multiplication with V^(-1/2)*U 
        where V is the diagonal matrix of eigenvalues sorted in
        decreasing order and U the corresponding eigenvectors.
        Reduces dimensionality if numComponents is less than
        data dimension.
        """

        # construct whitening matrix
        if numComponents == -1:
            numComponents = self.U.shape[0]

        W = np.dot(np.diag(np.sqrt(1/(self.V+self.epsilon))),self.U.T)
        W = W[:numComponents,:]
        return np.dot(W,data-self.meanPatch)

    def unwhiten(self,data):
        """
        Reconstructs original data from whitened data.
        """
        numComponents = data.shape[0]
        
        W = np.dot(self.U,(np.diag(np.sqrt(self.V+self.epsilon))))

        W = W[:,:numComponents]
        
        return W.dot(data)+self.meanPatch


    def plot_explained_var(self):
        """
        Plots variance of data explained as a function of the
        number of components.
        """
        import matplotlib.pyplot as plt
        plt.plot(range(1,self.U.shape[0]+1),np.cumsum(self.V/np.sum(self.V)))
        plt.show()
        
        
