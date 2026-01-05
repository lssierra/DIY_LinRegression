import numpy as np


class my_LinearRegression():
    def __init__(self):
        self.slopes_ = None
        self.coeff_ = 0.0

        self.epochs = 500
        self.learn_rate = 0.0001

    def Fit(self, X, y, alg='ClosedForm'):
        if alg == 'CloseForm':
            self.slopes_, self.coeff_ = self.ClosedForm(X,y)
        elif alg == 'GradDescent':
            for i in range(self.epochs):
                self.slopes_, self.coeff_ = self.GradDescent(X, y, self.slopes_, self.coeff_)
                if i%1000 == 0:
                print(i)
            
    def ClosedForm(self, X, y): 
        X = np.array(X)
        y = np.array(y)

        X = np.c_[np.ones((X.shape[0],1)),X]

        beta = np.linalg.inv(X.T @ X) @ X.T @ y

        self.slopes_ = beta[1:]
        self.coeff_ = beta[0]
        
    def GradDescent(self, X, y, m_, b_):
        X = np.array(X)
        y = np.array(y)

        pderiv_m = np.sum(-2 * (y - (X*m + b)) * X)/float(X.shape[0])
        pderiv_b = np.sum(-2 * (y - (X*m + b)))/float(X.shape[0])

        m_new = m - self.learn_rate*pderiv_m
        b_new = b - self.learn_rate*pderiv_b
        return m_new, b_new

    def Predict(self,X):
        X = np.array(X)

        return X @ self.slopes_ + self.coeff_       