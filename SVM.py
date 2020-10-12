""" 
Install pandas: pip3 install pandas
"""
import pandas as pd
import numpy as np
import random, math
from scipy . optimize import minimize 
import matplotlib . pyplot as plt
from sklearn.model_selection import KFold

class SVM:
    def __init__(self, C, sigma):
        self.test_data = None
        self.test_labels = None
        self.train_data = None
        self.train_labels = None
        self.N = None
        self.M = None
        self.nonZeroValues = []
        self.corrInputs = []
        self.corrTargets = []
        self.smallb = 0
        self.K = None
        self.start = None
        self.C = C
        self.B = None
        self.XC = None
        self.sigma = sigma
        self.num_support_vectors = 0

    def read_data_haberman(self,test_data, test_labels, train_data, train_labels):

        self.test_data = test_data
        self.test_labels = test_labels

        self.train_data = train_data
        self.train_labels = train_labels

        self.N = self.train_data.shape[0]
        self.M = self.test_data.shape[0]

    def read_data(self, test_data, test_labels, train_data, train_labels):
        # f = open(test_data, 'r')

        self.test_data = pd.read_csv(test_data, sep='  ', engine='python', header=None).to_numpy()
        self.test_labels = pd.read_csv(test_labels, sep='  ', engine='python', header=None).to_numpy()

        self.train_data = pd.read_csv(train_data, sep='  ', engine='python', header=None).to_numpy()
        self.train_labels = pd.read_csv(train_labels, sep='  ', engine='python', header=None).to_numpy()
        self.N = self.train_data.shape[0]
        self.M = self.test_data.shape[0]


    def plotData(self):
            index_pos_samples = np.where(self.test_labels > 0)[0]
            index_neg_samples = np.where(self.test_labels < 0)[0]

            classA = self.test_data[index_pos_samples]
            classB = self.test_data[index_neg_samples]
            plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
            plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
            xgrid=np.linspace(-5, 5)
            ygrid=np.linspace(-4, 4)
            grid=np.array([[self.indicatorD2(x, y) for x in xgrid] for y in ygrid])
            plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
            plt.axis('equal')
            plt.savefig('svmplot.pdf')
            plt.show()
           
    def createMatrix(self):
        for i in range(self.N):
            for j in range(self.N):
                self.K[i, j] = ((self.train_labels[i]*self.train_labels[j])*self.rbfKernel(self.train_data[i], self.train_data[j]))
        
    def rbfKernel(self, x, y):
        sqEukDist = np.linalg.norm(x-y)**2
        return math.exp(-(sqEukDist/(2*(self.sigma**2))))

    def objective(self, a):
        #return scalar value which should be minimized, eq 4
        alphaAndP = np.dot(a, self.K)
        return (0.5 * (np.dot(a, alphaAndP))) - np.sum(a)

    def zerofun(self, a):
        #return scalar value, calculates the value which should be constrained to zero
        return np.sum(np.dot(a, self.train_labels))

    def calculateB(self, s, sTarget, alpha):
        kernelSVandX = np.zeros(self.N)

        for i in range(len(self.corrInputs)):
            kernelSVandX[i] = self.rbfKernel(s, self.corrInputs[i])

        alphaTargets=alpha*self.train_labels
        s = 0
        for i in range(len(self.corrInputs)):
            s += self.nonZeroValues[i]*self.corrTargets[i]*kernelSVandX[i]
        return float(s - sTarget)

    #indicates new values
    def indicator(self, newPoint): 
        kernelValues = np.zeros(len(self.corrInputs))
        for i in range(len(self.corrInputs)):
            kernelValues[i] = self.rbfKernel(newPoint, self.corrInputs[i])

        s = 0
        for i in range(len(self.nonZeroValues)):
            s += self.corrTargets[i]*self.nonZeroValues[i]*kernelValues[i]
        
        if (s - self.smallb) > 0:
            return 1
        elif (s - self.smallb) < 0:
            return -1
        else:
            return 0
    
    def indicatorD2(self, newPoint_x, newPoint_y): 
        kernelValues = np.zeros(len(self.corrInputs))
        for i in range(len(self.corrInputs)):
            kernelValues[i] = self.rbfKernel((newPoint_x, newPoint_y), self.corrInputs[i])

        s = 0
        for i in range(len(self.nonZeroValues)):
            s += self.corrTargets[i]*self.nonZeroValues[i]*kernelValues[i]
        
        if (s - self.smallb) > 0:
            return 1
        elif (s - self.smallb) < 0:
            return -1
        else:
            return 0

    def calculateError(self):
        # svmResult = np.zeros(self.M)
        correct_guesses = 0
        for i in range(self.M):
            # svmResult[i] = self.indicator(self.test_data[i][0], self.test_data[i][1])
            if self.indicator(self.test_data[i]) == int(self.test_labels[i]):
                correct_guesses += 1
        # equalValues = np.sum(svmResult == self.test_labels)
        return (self.M - correct_guesses)/self.M

    # finds vector α⃗ which minimizes the function objective within the bounds B and the constraints XC.
    def findVector(self):
        self.K = np.zeros((self.N, self.N), dtype=float)
        #vector with the initial guess of the α⃗ vector
        self.start = np.zeros(self.N)
        #B is a list of pairs of the same length as the α⃗ -vector, stating the lower and upper bounds for the corresponding element in α⃗
        self.B = [(0, self.C) for b in range(self.N)]
        
        #equality constraint, the second half of (10)
        self.XC = {'type':'eq', 'fun':self.zerofun}

        self.createMatrix()

        ret = minimize(self.objective, self.start, bounds=self.B, constraints=self.XC )
        alpha = ret['x']
        foundSol = ret['success'] #will be true if the optimizer found a solution
        #print(foundSol)

        for i, alphaValue in enumerate(alpha):
            if np.abs(alphaValue) > 1e-6:
                self.nonZeroValues.append(alphaValue) #sparar alpha värdet
                self.corrInputs.append(self.train_data[i]) #tillsammans med corrresponding input och target
                self.corrTargets.append(self.train_labels[i])

        self.corrInputs = np.array(self.corrInputs)
        self.corrTargets = np.array(self.corrTargets)
        #threshold value b
        self.smallb = self.calculateB(self.corrInputs[0], self.corrTargets[0], alpha)
        #self.plotData()
        self.num_support_vectors = len(self.nonZeroValues)

if __name__ == "__main__":
    #C=3.162e+00; sigma=1.200e+02
    C_array= [0.1,0.2,0.3]
    sigma_array= [0.1, 0.5, 1, 1.5, 2, 2.5, 10]#[0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 2.0, 4.0]

    datasets = ["haberman"]#,"german","breast-cancer","banana"]
    
    # must be 1 for haberman
    N_datasets = 1

    # --- hyperparameters for banana
    # C=3.162e+02
    # sigma=1.000e+00
    # dataset = "banana"
   
    # --- hyperparameters for breast cancer
    # C = 15.19
    # sigma = 4
    # dataset = "breast-cancer"

    # --- hyperparameters for German
    # C=1.000e+09
    # sigma=1.500e+00
    # dataset = "ringnorm"
    # C=3.162e+00
    # sigma=5.500
    # dataset = "german"

    for dataset in datasets:

        print("\n\n####### DATASET : %s ########\n" % dataset)
        if dataset == "haberman":

            data = pd.read_csv("data/haberman_data.asc", sep=',', engine='python', header=None).to_numpy()
            labels = data[:,3]
            data = data[:,:2]

            labels[labels == 2] = -1.0 # if label = 2, patient DIED 
            labels[labels == 1] = 1.0
            
            kf = KFold(n_splits=5, shuffle=True, random_state=1337)
            best_error = 1.0
            best_C = 0
            best_sigma = 0

            for C in C_array:

                for sigma in sigma_array:

                    errorTot = 0
                    support_vectors_tot = 0

                    for train_index, test_index in kf.split(data):
                        train_data, test_data = data[train_index], data[test_index]
                        train_labels, test_labels = labels[train_index], labels[test_index]

                        svm = SVM(C, sigma)
                        svm.read_data_haberman(test_data, test_labels, train_data, train_labels)
                        svm.findVector()
                        errorRate = svm.calculateError()
                        errorTot += errorRate
                        support_vectors_tot += svm.num_support_vectors

                    averageError = errorTot/5
                    average_support_vectors = support_vectors_tot/5

                    if averageError < best_error:
                        best_error = averageError
                        best_C = C
                        best_sigma = sigma
                        best_sv = average_support_vectors
                    print("\tC= %f\tsig= %f\terr= %f\tSV:%f"%(C,sigma,averageError,average_support_vectors))
            
            print("\nBEST: \tC= %f\tsig= %f\terr= %f\tSV:%f"%(best_C,best_sigma,best_error,best_sv))
        
        else:

            for C in C_array:

                for sigma in sigma_array:
                
                    errorTot = 0
                    support_vectors_tot = 0

                    for i in range(N_datasets):
                        number = i+1

                        test_data = ("data/%s/%s_test_data_%i.asc" % (dataset, dataset, number))
                        test_labels = ("data/%s/%s_test_labels_%i.asc" % (dataset, dataset, number))

                        train_data = ("data/%s/%s_train_data_%i.asc" % (dataset, dataset, number))
                        train_labels = ("data/%s/%s_train_labels_%i.asc" % (dataset, dataset, number))

                        svm = SVM(C, sigma)
                        svm.read_data(test_data, test_labels, train_data, train_labels)
                        svm.findVector()
                        errorRate = svm.calculateError()
                        errorTot += errorRate
                        support_vectors_tot += svm.num_support_vectors

                    averageError = errorTot/N_datasets
                    average_support_vectors = support_vectors_tot/N_datasets

                    print("\tC= %f\tsig= %f\terr= %f\tSV:%f"%(C,sigma,averageError,average_support_vectors))




