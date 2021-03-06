import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


def get_clouds():
    Nclass=500
    D=2

    X1=np.random.randn(Nclass,D)+np.array([-2,2])
    X2=np.random.randn(Nclass,D)+np.array([2,2])
    X3=np.random.randn(Nclass,D)+np.array([0, -2])
    X = np.vstack([X1,X2,X3]).astype(np.float32)

    Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
    return X,Y

def init_weight(M1,M2):
    W=np.random.rand(M1,M2)/np.sqrt(M1)
    return W

def sigmoid(X, W,b):
    return 1/(1+np.exp(-X@W-b))


def relu (a):
    return [a>0]

def softmax(z, W, b):
    return np.exp(z@W+b)/np.exp(z@W+b).sum(axis=1,keepdims=True)

def cost_sigmoid(Y,T):
    return T.dot(np.log(Y))+(1-T).dot(np.log(1-Y))

def cost(T,Y):
    return -(T*np.log(Y)).sum()

def tanh(X,W,b):
    return ((np.exp(X@W + b)-np.exp(-X@W -b)))/((np.exp(X@W + b)+np.exp(-X@W -b)))

def error_rate(T,Y):
    return(Y!=T).mean()

def y2indicator(Y):
    N=len(Y)
    K=len(set(Y))
    ind=np.zeros((N,K))
    for i in range(N):
        ind[i,Y[i]]=1
    return ind

def get_data(balance_ones):
    X=[]
    Y=[]
    first=True
    for line in open('fer2013.csv'):
        if first:
            first=False
        else:
            row=line.split(',')
            if row[0].isdigit():
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

    X,Y=np.array(X)/255.0,np.array(Y)

    if balance_ones:
        #balance class one
        X0,Y0=X[Y!=1,:], Y[Y!=1]
        X1=X[Y==1,:]
        X1=np.repeat(X1,9,axis=0)
        X=np.vstack([X0,X1])
        Y=np.concatenate((Y0,[1]*len(X1)))

    return X, Y


def get_binary_data():
    X=[]
    Y=[]
    first=True
    for line in open('fer2013.csv'):
        if first:
            first=False
        else:
            row=line.split(',')
            if row[0].isdigit():
                y=int(row[0])
                if y==0 or y==1:
                    Y.append(y)
                    X.append([int(p) for p in row[1].split()])
    return np.array(X)/255.0, np.array(Y)

def get_image():
    X,Y=get_data()
    N,D=X.shape
    d=int(np.sqrt(D))
    X=X.reshape(N,1,d,d)
    return X,Y

def crossValidation(model, X,Y,K=5):
    X,Y=np.shuffle(X,Y)
    sz=len(X)//K
    errors=[]
    for k in range(K):
        xtrain=np.concatenate((X[:k*sz,:],X[(k+1)*sz:,:]))
        ytrain = np.concatenate((Y[:K*sz], Y[(k + 1) * sz:]))
        xtest=X[k*sz:(k+1)*sz,:]
        ytest=Y[k*sz:(k+1)*sz]

        model.fit(xtrain,ytrain)
        error=model.score(xtest,ytest)
        errors.append(error)
    print('errors:', errors)
    return np.mean(errors)

