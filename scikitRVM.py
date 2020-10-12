from skrvm import RVC
from sklearn.datasets import load_iris



if __name__ == "__main__":
    clf = RVC()
    clf.verbose = True # Print iteration, alpha, beta, gamma, m, Relevance vectors
    data = load_iris()
    trainData = data.data
    trainTargets = data.target
    print(clf.fit(trainData, trainTargets))
    #print(clf.score(trainData, trainTargets))
