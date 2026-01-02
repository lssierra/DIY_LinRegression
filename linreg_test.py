import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


### Make synthetic data

N_SAMPLES = 100
N_FEATS = 1


# # params
# noise = 10

# #Generate
# X, y = make_regression(N_SAMPLES, N_FEATS, noise=noise)
# print(X.shape,y.shape)

X = np.random.rand(N_SAMPLES)*100 -50


#params of line
m = 3
b = 1


y = X*m + b


#params of gaussian noise
center = 0
std = 10

noise = np.random.normal(center,std,N_SAMPLES)

#mix line and noise
y = y + noise

###

def loss_function(m, b, X, y):
    total_error = 0
    total_error = np.sum((y - (X*m + b))**2)
    mse = total_error/float(N_SAMPLES)

def grad_descent(m, b, X, y, learn_rate):
    pderiv_m = np.sum(-2 * (y - (X*m + b)) * X)/float(N_SAMPLES)
    pderiv_b = np.sum(-2 * (y - (X*m + b)))/float(N_SAMPLES)
    
    m_new = m - learn_rate*pderiv_m
    b_new = b - learn_rate*pderiv_b
    return m_new, b_new

###
m=1
b=1
epochs = 100000
L = 0.001


for i in range(epochs):
    m, b = grad_descent(m, b, X, y, L)
    if i%1000 == 0:
        print(i)

###
min_X = np.min(X)
max_X = np.max(X)

    

print(m,b)
plt.scatter(X,y,color = "black")
plt.plot(list(np.linspace(min_X,max_X,N_SAMPLES)), [m*x + b for x in np.linspace(min_X,max_X,N_SAMPLES)], color = "red") 
plt.show()



