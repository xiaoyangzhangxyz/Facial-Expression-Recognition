import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Components import softmax, tanh, relu, error_rate, y2indicator,cost,get_data

class ANN(object):
    def __init__(self,M):
        self.M=M


    def forward(self,X,W1,b1,W2,b2):
        Z=tanh(X,W1,b1)
        #z=relu(X,W1,b1)
        yp=softmax(Z,W2,b2)
        return Z, yp

    def fit(self,X,Y,learning_rate=10e-4,reg=10e-20,epochs=100000,show_fig=True):
        X,Y=shuffle(X,Y)
        x_train=X[1000:]
        y_train=Y[1000:]
        yt_ind=y2indicator(y_train)
        x_valid=X[:1000]
        y_valid=Y[:1000]
        yv_ind=y2indicator(y_valid)
        costs=[]
        best_validation_error=1

        N,D=X.shape
        K=len(set(Y))
        self.W1=np.random.randn(D,self.M)
        self.b1=np.zeros(self.M)
        self.W2=np.random.randn(self.M,K)
        self.b2=np.zeros(K)

        for i in range(epochs):
            z,yp=self.forward(x_train,self.W1,self.b1,self.W2,self.b2)
            self.W2-=learning_rate*(z.T@(yp-yt_ind)+reg*self.W2)
            self.b2-=learning_rate*((yp-yt_ind).sum()+reg*self.b2)
            self.W1-=learning_rate*(x_train.T@((yp-yt_ind)@self.W2.T*(1-z*z))+reg*self.W1)
            self.b1-=learning_rate*(((yp-yt_ind)@self.W2.T*(1-z*z)).sum()+reg*self.b1)

            if i%100==0:
                #cost
                zv,yvp=self.forward(x_valid,self.W1,self.b1,self.W2,self.b2)
                cost1=cost(yv_ind,yvp)
                costs.append(cost1)

                yvp_label=np.argmax(yvp,axis=1)
                error=error_rate(y_valid,yvp_label)
                if error < best_validation_error:
                    best_validation_error=error

                print('epochs:',i,'error of validation:',best_validation_error)

        if show_fig==True:
            plt.plot(costs)


    def predict(self,X):
        z, yp = self.forward(X, self.W1, self.b1, self.W2, self.b2)
        yp_label=np.argmax(yp,axis=1)

        return yp_label


    def score(self,X,Y):
        yp=self.predict(X)
        score=1-error_rate(Y,yp)
        return score

def main():
    X,Y=get_data(False)

    model=ANN(200)
    model.fit(X,Y)
    score=model.score(X,Y)
    print('score:', score)


if __name__=='__main__':
    main()






