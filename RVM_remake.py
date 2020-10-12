import pandas as pd
import numpy as np
import math
from scipy.special import gamma, factorial, expit
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class RVM:
    def __init__(self, sigma, tolerance):
        self.test_data = None
        self.test_labels = None
        self.train_data = None
        self.train_labels = None
        self.bias = True
        self.error_rate = 0
        self.sigma = sigma
        self.tolerance = tolerance

    def plotData(self, RV_X, RV_w):
        index_pos_samples = np.where(self.test_labels > 0.5)[0]
        index_neg_samples = np.where(self.test_labels < 0.5)[0]

        classA = self.test_data[index_pos_samples]
        classB = self.test_data[index_neg_samples]

        plt.plot([p[0] for p in RV_X], [p[1] for p in RV_X], 'gx')
        plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
        plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
        
        xgrid=np.linspace(-5, 5)
        ygrid=np.linspace(-4, 4)
        grid=np.array([[self.indicator((x, y), RV_X, RV_w) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, (0.0, 0.5, 1.0), colors=('red','black','blue'), linewidths=(1,2,1), zorder=10)
        plt.axis('equal')
        plt.savefig('rvmplot.pdf')
        plt.show()

    def read_data_haberman(self, test_data, test_labels, train_data, train_labels):
        
        self.test_data = test_data
        self.test_labels = test_labels
        self.train_data = train_data
        self.train_labels = train_labels

        self.N = self.train_data.shape[0]
        self.N_test = self.test_data.shape[0]

    def read_data(self, test_data, test_labels, train_data, train_labels):
        # f = open(test_data, 'r')
        self.test_data = pd.read_csv(test_data, sep='  ', engine='python', header=None).to_numpy()
        self.test_labels = pd.read_csv(test_labels, sep='  ', engine='python', header=None).to_numpy()

        self.train_data = pd.read_csv(train_data, sep='  ', engine='python', header=None).to_numpy()
        self.train_labels = pd.read_csv(train_labels, sep='  ', engine='python', header=None).to_numpy()

        # extra datasets that never gets pruned. for debugging.
        self.train_data2 = pd.read_csv(train_data, sep='  ', engine='python', header=None).to_numpy()
        self.train_labels2 = pd.read_csv(train_labels, sep='  ', engine='python', header=None).to_numpy()

        self.train_labels[self.train_labels == -1.0] = 0.0
        self.test_labels[self.test_labels == -1.0] = 0.0

        self.N = self.train_data.shape[0]
        self.N_test = self.test_data.shape[0]

    def rbf(self, x, y):
        sigma = self.sigma
        dist = np.power(np.linalg.norm(x-y), 2)
        return math.exp(-(dist/(2*(np.power(sigma, 2)))))

    def compute_phi(self):
        phi = np.zeros((self.N, self.N))
        for n in range(self.N):
            for j in range(self.N):
                phi[n,j] = self.rbf(self.train_data[n], self.train_data[j])

        phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)

        return phi

    # for classifying new samples
    # input: ddatapoint to classify, Relevance vectors, relevance weights
    def indicator(self, newPoint, RV_X, RV_w):
        # compute y
        if RV_w.shape[0] > RV_X.shape[0]:
            y = RV_w[-1] # add w0
        else:
            y = 0

        for i in range(RV_X.shape[0]):
            y += RV_w[i] * self.rbf(newPoint, RV_X[i]) 
        
        # compute probability/sigmoid
        
        prob = self.sigmoid(y)

        if prob >= 0.5:
            return 1.0
        elif prob <= 0.5:
            return 0.0
        
        return prob

    def sigmoid(self, y):
        return 1/(1+np.exp(-y))

    def y(self, x_index, w, Phi):
        return np.dot(w, Phi[x_index, :])
    
    def prune(self, w_mean, alphas, Phi, alphas_old):
        '''keep is a vector with True/False values depending on if we want to keep that data point or not'''
        keep = np.abs(alphas) < self.tolerance
        # increasing tolerance => fler RVs

        if not np.any(keep):
            keep[0] = True
            if self.bias:
                keep[-1] = True

        
        if self.bias:
            if not keep[-1]:
                self.bias = False
            self.train_data = self.train_data[keep[:-1]]
        else:
            self.train_data = self.train_data[keep]
        
        w_mean = w_mean[keep]
        alphas = alphas[keep]
        alphas_old = alphas_old[keep]
        Phi = Phi[:,keep]
        #Phi = Phi[keep[1:],:]
        
        self.N = self.train_data.shape[0]

        return w_mean, alphas, Phi, alphas_old

    def _classify(self, m, phi):
        return expit(np.dot(phi, m))

    def _log_posterior(self, m, alpha, phi, t):

        y = self._classify(m, phi)

        t = t.ravel()

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1-y[t == 0]), 0))
        log_p = log_p + 0.5*np.dot(m.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian

    def _hessian(self, m, alpha, phi, t):
        y = self._classify(m, phi)
        B = np.diag(y*(1-y))
        return np.diag(alpha) + np.dot(phi.T, np.dot(B, phi))

    def _posterior(self, w, alphas, Phi):
        result = minimize(
            fun=self._log_posterior,
            hess=self._hessian,
            x0=w,
            args=(alphas, Phi, self.train_labels),
            method='Newton-CG',
            jac=True,
            options={'maxiter': 50}
        )

        w = result.x
        cov = np.linalg.inv(
            self._hessian(w, alphas, Phi, self.train_labels)
        )
        return w, cov

    def createRVM(self):
        # intitialize alpha and w
        np.random.seed(1337)
        w = np.zeros(self.N+1)
        
        alphas = np.full(self.N+1, 0.1)
        alphas_old = alphas
        Phi = self.compute_phi()
        
        for iteration in range(4000):

            w, cov = self._posterior(w, alphas, Phi)

            gamma = 1 - alphas*np.diag(cov)
            alphas = gamma/(w ** 2)

            w, alphas, Phi, alphas_old = self.prune(w, alphas, Phi, alphas_old)

            #print('RVs: ', self.train_data.shape, "\niter: ", iteration)

            delta = np.amax(np.absolute(alphas - alphas_old))

            if delta < 1e-3 and iteration > 1:
                break
            
            alphas_old = alphas

        # self.plotData(self.train_data, w)

            #plotlist=[5,10,20,30,50]
            # if iteration in plotlist:
            #     self.plotData(self.train_data, w)
        
        correct_guesses = 0
        for i in range(self.N_test):
            indicated_value = self.indicator(self.test_data[i], self.train_data, w)
            if indicated_value == self.test_labels[i]:
                correct_guesses += 1
        
        self.error_rate = (self.N_test - correct_guesses ) / self.N_test
        self.relevance_vectors = self.train_data.shape[0]
        #print('RVs: ', self.train_data.shape, "\niter: ", iteration)

if __name__ == "__main__":

    sigma_vector = [1,2,3,4,5,6,7,8,9, 10,15]#[3.7,3.8,3.9,4.0,4.1,4.2,4.3,5.0]#[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    tolerance_vector = [1e12]

    datasets = ["haberman"]#["heart","german","banana","breast-cancer"]

    # N_datasets must be 1 for "haberman"
    N_datasets = 1

    for dataset in datasets:
        
        print("\n\n####### DATASET : %s ########\n" % dataset)
        if dataset == "haberman":

            data = pd.read_csv("data/haberman_data.asc", sep=',', engine='python', header=None).to_numpy()
            labels = data[:,3]
            data = data[:,:2]

            labels[labels == 2] = 0.0 # if label = 2, patient DIED 
            labels[labels == 1] = 1.0

            kf = KFold(n_splits=5, shuffle=True)

            for sigma in sigma_vector:

                for tolerance in tolerance_vector:

                    error_tot = 0
                    support_vectors_tot = 0

                    for train_index, test_index in kf.split(data):
                        train_data, test_data = data[train_index], data[test_index]
                        train_labels, test_labels = labels[train_index], labels[test_index]

                        rvm = RVM(sigma, tolerance)
                        rvm.read_data_haberman(test_data, test_labels, train_data, train_labels)
                        rvm.createRVM()
                                    
                        error_tot += rvm.error_rate
                        support_vectors_tot += rvm.relevance_vectors
                        break        
                    # error_tot /= 5
                    # support_vectors_tot /= 5

                    print("\ttol= %i\tsig= %f\terr= %f\tRV:%f"%(tolerance,sigma,error_tot,support_vectors_tot))

        else:
            for sigma in sigma_vector:

                for tolerance in tolerance_vector:

                    error_tot = 0
                    support_vectors_tot = 0
                    
                    for i in range(N_datasets):
                        number = i+1

                        test_data = ("data/%s/%s_test_data_%i.asc" % (dataset, dataset, number))
                        test_labels = ("data/%s/%s_test_labels_%i.asc" % (dataset, dataset, number))

                        train_data = ("data/%s/%s_train_data_%i.asc" % (dataset, dataset, number))
                        train_labels = ("data/%s/%s_train_labels_%i.asc" % (dataset, dataset, number))

                        rvm = RVM(sigma, tolerance)
                        rvm.read_data(test_data, test_labels, train_data, train_labels)
                        rvm.createRVM()
                        
                        error_tot += rvm.error_rate
                        support_vectors_tot += rvm.relevance_vectors
                    
                    error_tot /= N_datasets
                    support_vectors_tot /= N_datasets

                    print("\ttol= %i\tsig= %f\terr= %f\tRV:%f"%(tolerance,sigma,error_tot,support_vectors_tot))

