import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
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
    epochs = 50
    mnnetTrain = NN(X_train, Y_train)
    global accuracy
    accuracyTrain = []
    for z in range(epochs):
    #for z in range(0,1): #For the epoch...
        global acc
        acc = 0
        dindex = 0
        print("Epoch:", z)
        #for i in range(len(X_train)):  #For the 60K data
        for i in X_train:
            #print("Data #:", i)
            mnnetTrain.propForward(i,  dindex)
            mnnetTrain.backprop(i)
            dindex +=1


        acc = mnnetTrain.accuracy(X_train, Y_train)
        print("Accuracy for this epoch:", acc)
        accuracyTrain.append(acc)

    #acc2 =0
    accuracyTest = []
    for z in range(epochs):
        acc2 = 0
        print("Epoch for testing:", z)
        acc2 = mnnetTrain.accuracy(X_test,Y_test)
        print("Accuracy for this epoch", acc2)
        accuracyTest.append(acc2)




    # print(predictions)
    # print(Y_train)
    # cm = confusion_matrix(Y_train,predictions)
    # print("Confusion Matrix:")
    # print(cm)
        #Shuffle data before next epoch?
        #permutation = np.random.permutation(X_train.shape[1])
        #X_train_shuffle = X_train[:,permutation]
        #Y_train_shuffle = Y_train[:,permutation)

        #OR SHUFFLE LIKE THIS:
        #m= 500 #number of data in training or testing?
        #shuffle = np.random.permutation(m)
        #X_train, Y_train = X_train[:,shuffle], Y_train[:, shuffle]


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



    def createMatrices(self,numInputs, numHidden, outNum):


        self.weightsJI = np.random.uniform(-0.5, 0.5, size=(numInputs, numHidden))
        self.weightsKJ = np.random.uniform(-0.5, 0.5, size=(numHidden, outNum))
        self.bias1 = np.ones((1, numHidden))
        self.bias2 = np.ones((1, outNum))
        # print("Bias1:", self.bias1)
        # print("Bias2: ", self.bias2)

        self.deltaChangeK = np.zeros(len(self.weightsKJ))
        self.deltaChangeJ = np.zeros(len(self.weightsJI))

# Propogate input forward
    def propForward(self,data, dataindex) :
        global tK

        tK = [0.1 for j in range(10)]
        tK[self.y[dataindex]] = 0.9

        z = np.dot(data, self.weightsJI) + self.bias1
        self.hiddenL = sigmAct(z)
        d = np.dot(self.hiddenL, self.weightsKJ) + self.bias2
        self.outputs = sigmAct(d)
        return self.outputs
        # print("HiddenLayers: ", self.hiddenL)
        # print("Outputs: ", self.outputs)

    def backprop(self,data ):
        deltaK, deltaJ = self.errorcalc()
        # print("This is deltaK", deltaK)
        # print("This is deltaJ", deltaJ)
        #update weights w/ momentum

        self.deltaChangeK = (self.lr * deltaK * self.hiddenL) + (self.mom *self.deltaChangeK)
        self.weightsKJ = self.weightsKJ + self.deltaChangeK.T
        #print(self.weightsKJ)

        self.deltaChangeJ = (self.lr * deltaJ * data) +(self.mom *self.deltaChangeJ)
        self.weightsJI = self.weightsJI + self.deltaChangeJ.T
        # print(self.weightsJI)

    def accuracy(self, dataX, dataY):
        #global predictions
        predictions = []
        index = 0
        for xx, yy in zip(dataX, dataY):

            out = self.propForward(xx,index)
            pred = np.argmax(out)
            predictions.append(pred==yy)
            index += 1
        summed = sum(pred for pred in predictions)/100
        return np.average(summed)

# Caluclate error
    def errorcalc(self):

        dK = (1- self.outputs)*(tK - self.outputs)
        deltaK = np.dot(self.outputs, dK.T)

        dJ = (1-self.hiddenL) * np.sum(self.weightsKJ.T * deltaK)
        deltaJ = np.dot(self.hiddenL, dJ.T)
        return deltaK, deltaJ


main()
