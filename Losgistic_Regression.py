import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Components import sigmoid, softmax, get_binary_data, error_rate


class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self,X,Y,learning_rate=10e-4,reg=10e-22,epoch=1200,fig_show=False):
        X,Y=shuffle(X,Y)
        x_train=X[1000:]
        y_train=Y[1000:]
        x_valid=X[:1000]
        y_valid=Y[:1000]

        N,D =X.shape
        self.W=np.random.randn(D)/np.sqrt(D)
        self.b=0
        errors=[]
        best_validation_error=1
        for i in range(epoch):
            y_pred=self.forward(x_train)
            self.W-=learning_rate*(x_train.T.dot((y_pred-y_train))+reg*self.W)
            self.b-=learning_rate*((y_pred-y_train).sum()+reg*self.b)

            if i%20==0:
                yv_pred=np.round(self.forward(x_valid))
                #print(yv_pred, self.W)
                error=error_rate(y_valid,yv_pred)
                if error < best_validation_error:
                    best_validation_error=error
                print('i:', i,'error:',best_validation_error)





    def forward(self,X):
        return sigmoid(X,self.W,self.b)

    def predict(self,X):
        return np.round(self.forward(X))

def main():
    X,Y=get_binary_data()
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0] * len(X0) + [1] * len(X1))
    #(X[:5],Y[:5])
    model=LogisticModel()
    model.fit(X,Y)
    prediction=model.predict(X)
    error=error_rate(Y,prediction)

if __name__=='__main__':
    main()





