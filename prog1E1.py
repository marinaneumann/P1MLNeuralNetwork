import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
# visualization tools
import matplotlib.pyplot as plt

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

# Termination condition (epochs)
def build(X_train, Y_train, X_test, Y_test):
    #X_train = train_test_split(X_train, shuffle=True)  # NEW DATASET HAS split training test data into 2  arrays
    #NEED to split data into smaller dataset & shuffle??

    z = 0  #for iteration in epoch
    epoch = 50
    N = input("How many hidden layers?")
    M = input("What momentum? For default, enter 0.9")
    mnnetTrain = NN(X_train, Y_train, N, M)

    #for z in range(epochs):
    mnnetTrain.propforward()
        #mnnetTrain.backprop()

        #Shuffle data before next epoch?

        #NEED TO DO SOMETHING WITH ACCURACY counting or something???

class NN:

 #initalize all weights
    def __init__(self,x, y, N, M):
        self.x = x  #input values from data
        self.inputs = 784
        self.numHidden = N #allowing for user to choose num hiddenL
        self.outs = 10
        self.lr = 0.1
        self.mom = M   #allowing for user to choose momentum, by default will be 0.9
        self.y = y  #labels from input for kth class
        inp_dim = x.shape[1]
        out_dim = y.shape[0]

        self.weightsJI = np.random.uniform(-0.5,0.5, size =784)
        self.weightsKJ = np.random.uniform(-0.5,0.5 ,size =50)
        self.bias1 = np.ones((1,50))
        self.bias2 = np.ones((1,10))

        self.deltaChangeK = np.zeros(784)
        self.deltaChangeJ = np.zeros(784)

# Propogate input forward
    def propforward(self) :
        z = np.dot(X_train[0], self.weightsJI) + self.bias1
        self.hiddenL = sigmAct(z)
        d = np.dot(self.hiddenL, self.weightsKJ) + self.bias2
        self.outputs = sigmAct(d)
        print("HiddenLayers: ", self.hiddenL)
        print("Outputs: ", self.outputs)

    def backprop(self):
        deltaK, deltaJ = 0
        deltaK, deltaJ = errorcalc(self,deltaK, deltaJ)
        print("This is deltaK", deltaK)
        print("This is deltaJ", deltaJ)
        #update weights w/ momentum

        # deltaChangeK = (self.lr * delaK * self.hiddenL) + (self.mom *self.deltaChangeK)
        # self.weightsKJ = self.weightsLJ + deltaChangeK
        #
        # deltaChangeJ = (self.lr * deltaJ * self.inputs) +(self.mom *self.deltaChangeJ)
        # self.weightsJI = self.weightsJI + deltaChangeJ

# Caluclate error
def errorcalc(self, deltaK, deltaJ):
    global tK
    tK= []
    #Figures out tK values depending on if input class is kth class or not.. don't think this is right
    for i in self.outputs:
        if self.outputs[i] == Y_train[i]:
            tK = 0.9
        else:
            tK = 0.3

    deltaK = self.outputs * (1- self.outputs)* (tK - self.outputs)
    deltaJ = self.hiddenL * (1- self.hiddenL) * np.dot(self.weightsKJ, deltaK)

    return deltaK, deltaJ


main()