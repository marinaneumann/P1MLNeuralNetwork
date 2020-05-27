import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
# visualization tools
#import matplotlib.pyplot as plt

def main():
    print("Neural Network For MNISET data")
    print("By Marina Neumann")
    print('CS445 Spring2020')
    print("Programming Assignment #1")


    dataLoad() #Load data from files
    build(X_train, Y_train, X_test, Y_test)     #build NN and run epochs

#Data data --> MNISET
def dataLoad():
    global X_train, Y_train,  X_test, Y_test

    #Code used to read in and create csv files.Becomes unnecessary after initial run.
    # X_test, Y_test = loadlocal_mnist(images_path='data/t10k-images-idx3-ubyte',labels_path='data/t10k-labels-idx1-ubyte')
    # np.savetxt(fname='data/testimages.csv',X=X_test, delimiter=',', fmt='%d')
    # np.savetxt(fname='data/testlabels.csv', X=Y_test, delimiter=',', fmt='%d')
    # X, Y = loadlocal_mnist(images_path='data/train-images-idx3-ubyte', labels_path='data/train-labels-idx1-ubyte')
    # np.savetxt(fname='data/trainimages.csv', X=X, delimiter=',', fmt='%d')
    # np.savetxt(fname='data/trainlabels.csv', X=Y, delimiter=',', fmt='%d')

    X = np.genfromtxt('data/trainimages.csv', delimiter=',', dtype= float)
    Y = np.loadtxt('data/trainlabels.csv', delimiter=',', dtype=int)
    X_train = X / 255.0
    Y_train = Y

    M = np.genfromtxt('data/testimages.csv', delimiter=',', dtype=float)
    N = np.loadtxt('data/testlabels.csv', delimiter=',', dtype=int)
    X_test = M/255.0
    Y_test = N

def sigmAct(s):
   return 1/(1+np.exp(-s))

def build(X_train, Y_train, X_test, Y_test):
    #X_train = train_test_split(X_train, shuffle=True)  # NEW DATASET HAS split training test data into 2  arrays
    #NEED to split data into smaller dataset & shuffle??
    #Experiment 3 does 2 separate datasets anyway... hmmm

    z = 0  #for iteration in epoch
    epoch = 50
    mnnetTrain = NN(X_train, Y_train)

    #for z in range(epochs):
    for z in range(0,1):
        mnnetTrain.propForward()
        #mnnetTrain.backprop()

        #Shuffle data before next epoch?
        #permutation = np.random.permutation(X_train.shape[1])
        #X_train_shuffle = X_train[:,permutation]
        #Y_train_shuffle = Y_train[:,permutation)

        #OR SHUFFLE LIKE THIS:
        #m= 500 #number of data in training or testing?
        #shuffle = np.random.permutation(m)
        #X_train, Y_train = X_train[:,shuffle], Y_train[:, shuffle]
        #NEED TO DO SOMETHING WITH ACCURACY counting or something???

class NN:

 #initalize all weights
    def __init__(self,x,y):
        self.x = x  #input values from data
        self.y = y  # labels from input for kth class
        numInputs = 784
        numHidden = int(input("How many hidden layers?(ie. 50, 100, 150...etc. "))
        M = float(input("What momentum? For a default choice please enter 0.9 :"))
        outNum = 10
        self.lr = 0.1
        self.mom = M   #allowing for user to choose momentum, by default will be 0.9
        self.createMatrices(numInputs, numHidden, outNum)

        # self.deltaChangeK = np.ones()
        # self.deltChangeJ = np.ones()

    def createMatrices(self,numInputs, numHidden, outNum):
        # inp_dim = x.shape[1]
        # out_dim = y.shape[0]

        self.weightsJI = np.random.uniform(-0.5, 0.5, size=(numInputs, numHidden))
        self.weightsKJ = np.random.uniform(-0.5, 0.5, size=(numHidden, outNum))
        self.bias1 = np.ones((1, numHidden))  # might need to change biases??
        self.bias2 = np.ones((1, outNum))
        # print("Bias1:", self.bias1)
        # print("Bias2: ", self.bias2)

        self.deltaChangeK = np.zeros(len(self.weightsKJ))
        self.deltChangeJ = np.zeros(len(self.weightsJI))

# Propogate input forward
    def propForward(self) :
        global tK
        #tK = np.zeros(10, dtype=float)
        tK = [[0.1 for j in range(10)] for i in range(len(self.y))]
        #print("The target values before assignment:", tK)
        index = 0
        for z in tK:
            z[self.y[index]] = 0.9
            #print(z)
            index +=1




        z = np.dot(self.x, self.weightsJI) + self.bias1
        self.hiddenL = sigmAct(z)
        d = np.dot(self.hiddenL, self.weightsKJ) + self.bias2
        self.outputs = sigmAct(d)
        # print("HiddenLayers: ", self.hiddenL)
        # print("Outputs: ", self.outputs)

    def backprop(self):
        deltaK, deltaJ = self.errorcalc()
        print("This is deltaK", deltaK)
        print("This is deltaJ", deltaJ)
        #update weights w/ momentum

        deltaChangeK = (self.lr * deltaK * self.hiddenL) + (self.mom *self.deltaChangeK)
        self.weightsKJ = self.weightsLJ + deltaChangeK

        deltaChangeJ = (self.lr * deltaJ * self.inputs) +(self.mom *self.deltaChangeJ)
        self.weightsJI = self.weightsJI + deltaChangeJ


# Caluclate error
    def errorcalc(self):

        deltaK = self.outputs * (1- self.outputs)* (tK - self.outputs)
        deltaJ = self.hiddenL * (1- self.hiddenL) * np.dot(self.weightsKJ, deltaK)

        return deltaK, deltaJ

    def accuracyTesting(self):
        count = 0
        for i in self.outputs:
            if i == self.y:
                count +=1

        return (count/len(self.x))*100
main()
