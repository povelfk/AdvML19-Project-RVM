import pandas as pd
import numpy as np
import math
from scipy.special import gamma, factorial
import sys

class RVM:
    def __init__(self):
        self.test_data = None
        self.test_labels = None
        self.train_data = None
        self.train_labels = None
        self.M = None

    def read_data(self, test_data, test_labels, train_data, train_labels):
        # f = open(test_data, 'r')
        self.test_data = pd.read_csv(test_data, sep='  ', engine='python', header=None)
        self.test_labels = pd.read_csv(test_labels, sep='  ', engine='python', header=None)

        self.train_data = pd.read_csv(train_data, sep='  ', engine='python', header=None).to_numpy()
        self.train_labels = pd.read_csv(train_labels, sep='  ', engine='python', header=None).to_numpy()

        self.train_labels[self.train_labels == -1] = 0

        self.N = self.train_data.shape[0]
        self.M = self.test_data.shape[0]
    '''
    def plotData(self):
            index_pos_samples = numpy.where(self.test_labels > 0.5)[0]
            index_neg_samples = numpy.where(self.test_labels < 0.5)[0]

            classA = self.test_data[index_pos_samples]
            classB = self.test_data[index_neg_samples]
            plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
            plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
            xgrid=numpy.linspace(-5, 5)
            ygrid=numpy.linspace(-4, 4)
            grid=numpy.array([[self.sigmoidD2(x, y) for x in xgrid] for y in ygrid])
            plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
            plt.axis('equal')
            plt.savefig('rvmplot.pdf')
            plt.show()
    '''


    def rbf(self, x, y):
        sigma = 1
        dist = np.power(np.linalg.norm(x-y), 2)
        return math.exp(-(dist/(2*(np.power(sigma, 2)))))

    def compute_phi(self):
        phi = np.ones((self.N, self.N+1))
        for n in range(self.N):
            for j in range(1,self.N+1):
                phi[n,j] = self.rbf(self.train_data[n], self.train_data[j-1])

        return phi

    def sigmoid(self, y):
        return 1/(1+np.exp(-y))
    
    def sigmoidIndicator(self, y):
        w = self.wMP
        if probability > 0.5:
            return 1
        else:
            return -1

    def y(self, x_index, w, Phi):
        temp = np.dot(w, Phi[x_index, :])
        return temp
    
    def calculateError(self, indicatedValues):
        correct_guesses = 0
        for i in range(self.M):
            if indicatedValues[i] == self.test_labels[i]:
                correct_guesses += 1
        return (self.M - correct_guesses)/self.M

    def createRVM(self):
        # intitialize alpha and w
        w = np.random.normal(0, 0.1, self.N+1)
        alphas = np.full(self.N+1, 0.1)
        Phi = self.compute_phi()

        # compute mean and covariance of the posterior
        for i in range(100):
            A = np.identity(self.N+1) * alphas
            B = np.zeros((self.N, self.N))
            
            Y = np.zeros(self.N)
            for n in range(self.N):
                Y[n] = self.sigmoid(self.y(n, w, Phi))
                B[n,n] = Y[n] - (1-Y[n])

            cov = np.linalg.inv(np.matmul(Phi.T, np.matmul(B, Phi) ) + A)

            Bt = np.matmul(B, self.train_labels.ravel())   
            SigmaPhi = np.matmul(cov, Phi.T)   
            
            w_mean = np.matmul(SigmaPhi, Bt)

            # update alpha
            for i, alpha in enumerate(alphas):
                if np.power(w_mean[i],2) < sys.float_info.epsilon:
                    continue    
                gamma_i = 1 - (alpha * cov[i,i])
                new_alpha = gamma_i / np.power(w_mean[i],2)
                alphas[i] = new_alpha
            
            w = w_mean

        errorRate = self.calculateError()
        temp = np.where(w > 1e-6)[0]

        print("RVs: ",temp.shape)
        print("ERROR: ", errorRate)
if __name__ == "__main__":

    dataset = "banana"
    errorTot = 0
    support_vectors_tot = 0
    
    for i in range(1):
        number = i+1

        test_data = ("data/%s/%s_test_data_%i.asc" % (dataset, dataset, number))
        test_labels = ("data/%s/%s_test_labels_%i.asc" % (dataset, dataset, number))

        train_data = ("data/%s/%s_train_data_%i.asc" % (dataset, dataset, number))
        train_labels = ("data/%s/%s_train_labels_%i.asc" % (dataset, dataset, number))

        rvm = RVM()
        rvm.read_data(test_data, test_labels, train_data, train_labels)
        rvm.createRVM()
        