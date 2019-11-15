'''Own implementation of Fully connected Neural Network'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1234)

def read_data():
    '''Reads data from file and returns training and test sets'''
    data = pd.read_csv("2016ME20781.csv",header=None)
    data = data.values.astype('float')
    x = data[:,:-1]/255.0
    N = len(data)
    x = np.hstack((np.ones((N,1)),x))
    y = data[:,-1]
    y = y.astype('int')
    
    # Splitting data into training and test set
    x_train = x[:2400,:]
    y_train = y[:2400]
    x_test = x[2400:,:]
    y_test = y[2400:]
    return x_train,y_train,x_test,y_test

def softmax(a):
    '''Softmax activation function to be used in output layer'''
    b = np.exp(a)
    s = sum(b)
    return b/s

def sigmoid(num):
    '''Returns the sigmoid of a number/array/matrix'''
    return 1/(1+np.exp(-num))

def sigmoidGradient(x):
    '''Return the derivative of sigmoid of a number/array/matrix'''
    return sigmoid(x)*(1-sigmoid(x))

def cost(Y,H):
    '''Returns the error of the model'''
    J = np.mean(Y*np.log(H)+(1-Y)*np.log(1-H))
    return -J

def accuracy(Y,H):
    '''Returns the accuracy of the model'''
    match=0
    h = np.argmax(H, axis=1).T
    for i in range(len(Y)):
        if(Y[i] == h[i]):
            match += 1
    return (match/len(Y))*100

def initializeWeights(l,N):
    '''Randomly initialize weights with mean 0'''
    w=[]
    for i in range(l):
        w.append(np.random.normal(0,N[i]**-0.5,[N[i],N[i+1]]))
    return w

def oneHotEncoding(y):
    '''Converts Label encoding to one hot encoding'''
    n = len(y)
    Y = np.zeros((n,10), dtype=int)
    for i in range(n):
        Y[i,y[i]] = 1
    return Y

def predict(w,l,x_test,activation):
    '''Returns the predicted output for given weights and input'''
    z = []
    a = []
    a.append(np.dot(x_test, w[0]))
    z.append(sigmoid(a[0]))
    if(l>1):
        i = 1
        while(i < l-1):
            a.append(np.dot(z[i-1], w[i]))
            z.append(sigmoid(a[i]))
            i += 1
        a.append(np.dot(z[i-1], w[i]))
        if(activation=="sigmoid"):
            z.append(sigmoid(a[i]))
        elif(activation=="softmax"):
            z.append(softmax(a[i].T).T)
    return z,a

def findWeights(w,l,x_train,y_train,alpha,batch,epoch,lam,activation):
    '''Finds the optimum weights using Gradient descent
    alpha = Learning rate, batch = batch_size
    epoch = maximum number of epoches, lam = lambda(regularisation)
    w1 = weights, activation = activation used in output layer'''
    J = []
    acc=[]
    n = len(y_train)
    for j in range(epoch):
        p=0
        while(p < n):
            x = x_train[p:p+batch,:]
            y = y_train[p:p+batch]
            Y = oneHotEncoding(y)
            z,a = predict(w,l,x,activation)
            J.append(cost(Y,z[l-1]))
            acc.append(accuracy(y, z[l-1]))
            p = p+batch
            delta = []
            delta.append((z[l-1]-Y))
            i = l-1
            while(i > 0):
                i -= 1
                e1 = np.dot(delta[l-i-2],w[i+1].T)
                e2 = sigmoidGradient(a[i])
                delta.append(e1*e2)
            w[0] = (1-lam*alpha)*w[0]-alpha*np.dot(x.T,delta[l-1])/batch
            i=1
            while(i<l):
                w[i] = (1-lam*alpha)*w[i]-alpha*np.dot(z[i-1].T,delta[l-i-1])/batch
                i += 1
    plt.figure()
    plt.plot(J)
    plt.figure()
    plt.plot(acc)
    return w

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = read_data()
    Y = oneHotEncoding(y_test)
    [n,m] = x_train.shape
    [n,l] = Y.shape
    
    # Setting hyperparameters
    #N = [m,100,l]
    #layerNum = 2
    #N = [m,250,100,50,l]
    #layerNum = 4
    N = [m,100,50,l]         # Number of units in each layer
    layerNum = 3             # Number of layers
    alpha = 0.3              # Learning rate
    batch = 100              # Batch size
    epoch = 100              # Number of epoch
    lam = 0.001                  # Lambda (Regularisation)
    
    w = initializeWeights(layerNum,N)
    
    # With sigmoid activation function in output layer
    w = findWeights(w,layerNum,x_train,y_train,alpha,
                    batch,epoch,lam,activation="sigmoid")    
    z,a = predict(w,layerNum,x_test,activation="sigmoid")
    h = z[layerNum-1]
    acc = accuracy(y_test,h)
    J = cost(Y,h)
    print("With sigmoid activation function in output layer")
    print("Cross Validation accuracy of the model: ",acc)
    print("Cross Validation cost of the model: ",J)
    print("---------------------------------------------------------")
    
    # With softmax activation function in output layer
    w = findWeights(w,layerNum,x_train,y_train,alpha,
                    batch,epoch,lam,activation="softmax")    
    z,a = predict(w,layerNum,x_test,activation="softmax")
    h = z[layerNum-1]
    acc = accuracy(y_test,h)
    J = cost(Y,h)
    print("With softmax activation function in output layer")
    print("Cross Validation accuracy of the model: ",acc)
    print("Cross Validation cost of the model: ",J)
