import numpy as np


class my_LinearRegression():
    def __init__(self):
        self.more_than_1D = 1

        self.slopes_ = None
        self.coeff_ = 0.0

        self.epochs = 500
        self.learn_rate = 0.0001


    def CheckShape(self,X):
        X = np.asarray(X)
        if X.ndim == 0:
            print("X must be at least 1-D")
            return 0
        elif X.ndim == 1:
            self.more_than_1D = 0
        else:
            self.more_than_1D = 1
        
        return X
        





    def Fit(self, X, y, alg='ClosedForm'):
        X = self.CheckShape(X)
        y = np.asarray(y)

        if alg == 'ClosedForm':
            self.ClosedForm(X,y)
        elif alg == 'GradDescent':
            for i in range(self.epochs):
                self.slopes_, self.coeff_ = self.GradDescent(X, y, self.slopes_, self.coeff_)
                if i%100 == 0:
                    print(i)
            
    def ClosedForm(self, X, y): 

        X = np.asarray(np.c_[np.ones((X.shape[0],1)),X])

        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

        if self.more_than_1D == 0:
            self.slopes_ = beta[1]
        else:
            self.slopes_ = np.asarray(beta[1:])
        self.coeff_ = beta[0]
        
    def GradDescent(self, X, y, m_, b_):

        if self.more_than_1D == 0:
            if m_ == None:
                m_ = 0
            #Prediction
            y_ = X * m_ + b_
        elif self.more_than_1D == 1:
            if m_ == None:
                m_ = np.zeros(X.shape[1])
            #Prediction
            y_ = X @ m_ + b_

        

        #Error vector
        error = y - y_

        #Gradient
        d_db = np.sum(-2 * error)/float(X.shape[0])
        d_dm = (-2 * X.T @ error)/float(X.shape[0])

        #New values
        m_ -= self.learn_rate*d_dm
        b_ -= self.learn_rate*d_db
        return m_, b_

    def Predict(self,X):
        X = self.CheckShape(X)
        

        if self.more_than_1D == 0:
            return X * self.slopes_ + self.coeff_
        else:
            return X @ self.slopes_ + self.coeff_

      