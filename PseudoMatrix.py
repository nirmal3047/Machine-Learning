'''Linear regression using gradient descent'''
import numpy as np
from matplotlib import pyplot as plt

# Reading file and loading data
data = np.loadtxt(open("Gaussian_noise.csv", "r"), delimiter = ",")
x = data[:, :-1]
y = data[:, 1:]

# Normalizing data
x = x/max(x)
y = y/max(y)

def featureMatrix(x, m):
    '''Return feature matrix of size [n,m]'''
    n = x.size
    I = np.ones((n,1))
    phi = I
    for i in range(1,m+1):
        phi = np.concatenate((phi, x**i), axis=1)
    return phi

def crossValidationError(x, y, m, l, alpha, iterNum):
    '''Returns cross Validation error for given degree of polynomial'''
    n = y.size
    u = int(n/4)
    J_cv = []
    for i in range(4):
        x_train = np.concatenate((x[:u*i], x[u*(i+1):]), axis=0)
        y_train = np.concatenate((y[:u*i], y[u*(i+1):]), axis=0)
        x_cv = x[i*u:(i+1)*u]
        y_cv = y[i*u:(i+1)*u]
        phi_train = featureMatrix(x_train, m)
        phi_cv = featureMatrix(x_cv, m)
        theta = theta_gradientDescent(phi_train, y_train, alpha, iterNum, 0)
        J_cv.append(cost(phi_cv, y_cv, theta, l))
    return sum(J_cv)

def cost(phi,y,theta,l):
    '''Returns regularised error corrosponding to given weight and lambda
    l=lambda (Regularizing coefficient)'''
    n = x.size               # Number of data points
    h = np.dot(phi,theta)
    error = 0.5*(sum((h-y)**2))/n + (l/2)*sum(theta**2)/n
    return error

def theta_gradientDescent(x, y, m, alpha, iterNum, lambd, epsilon, batch):
    '''Returns theta using Gradient Descent method
    alpha = Learning rate, iterNum = max number of iterations'''
    theta = np.random.rand(m+1,1)
    i = 0
    J = [2,1]
    L = len(y)
    while (abs(J[i+1]-J[i])>epsilon):
        if(i<iterNum):
            x_current = x[(i*batch)%L:(i*batch)%L+batch]
            y_current = y[(i*batch)%L:(i*batch)%L+batch]
            phi = featureMatrix(x_current, m)
            phi_t = np.transpose(phi)
            h = np.dot(phi, theta)
            theta = theta-alpha*np.dot(phi_t,(h-y_current))
            J.append(cost(phi,y_current,theta,0))
            i = i+1
        else:
            break
    t = np.arange(i+2)
    plt.figure()
    plt.ylim(0,0.002)
    plt.scatter(t[1000:],J[1000:])
    return theta

def useGradientDescent():
    '''Linear regression using gradient descent'''
    m = 18
    lamb = 0.011
    phi = featureMatrix(x, m)
    thetaGradient = theta_gradientDescent(x, y, m, 0.015, 10**6, lamb, 10**-8, 10)
    print('Weights found by Gradient Descent:\n', thetaGradient)
    h = np.dot(phi, thetaGradient)
    J = cost(phi, y, thetaGradient, lamb)
    print('Mean square Error of this model is: ', J)
    plt.figure()
    plt.title('Scatter plot')
    plt.scatter(x, y)
    plt.scatter(x, h)
    plt.legend(('Actual Data', 'Predicted Fit'), loc='upper right')

if __name__ == '__main__':
    useGradientDescent()
