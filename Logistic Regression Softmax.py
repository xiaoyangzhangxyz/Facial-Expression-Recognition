import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Components import softmax, error_rate, y2indicator, cost, get_data
class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self,X,Y,learning_rate=10e-8,reg=10e-20,epochs=1000,show_fig=False):
        X, Y = shuffle(X, Y)
        x_train=X[1000:]
        y_train=Y[1000:]
        x_valid=X[:1000]
        y_valid=Y[:1000]

        N,D=X.shape
        K=len(set(Y))
        self.W=np.random.randn(D,K)/np.sqrt(K)
        self.b=np.random.randn(K)


        best_validation_error=1
        costs=[]

        for i in range(epochs):
            y_soft=softmax(x_train,self.W,self.b)
            yt_ind=y2indicator(y_train)
            self.W-=learning_rate*(x_train.T@(y_soft-yt_ind)+reg*self.W)
            self.b-=learning_rate*((y_soft-yt_ind).sum()+reg*self.b)


            if i%20==0:
                yv_soft=softmax(x_valid,self.W,self.b)
                #cost
                yv_ind=y2indicator(y_valid)
                cost1=cost(yv_ind,yv_soft)
                costs.append(cost1)
                #error
                yv_pred=np.argmax(yv_soft,axis=1)
                error=error_rate(y_valid,yv_pred)
                if error<best_validation_error:
                    best_validation_error=error

                print("error of validation:", best_validation_error)

        if show_fig:
            plt.plot(costs)

    def predict(self,X):
        Y_soft = softmax(X, self.W, self.b)
        Y_pred = np.argmax(Y_soft, axis=1)
        return Y_pred

    def accuracy(self,X,Y):
        Y_pred=self.predict(X)
        return (1-error_rate(Y,Y_pred))


def main():
    X,Y=get_data(False)

    model=LogisticModel()
    model.fit(X,Y)
    accuracy=model.accuracy(X,Y)
    print('accuracy:',accuracy)


if __name__=='__main__' :
    main()


