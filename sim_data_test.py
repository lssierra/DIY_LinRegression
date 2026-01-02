import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


N_SAMPLES = 100
N_FEATS = 1


### Using sklearn to generate simulated data 

# params
noise = 10

#Generate
X, y = make_regression(N_SAMPLES, N_FEATS, noise=noise)


#plot
fig, ax = plt.subplots()
ax.scatter(X,y)
# plt.show()


### Using a ramdom points sampled uniformle from a line and adding gaussian noise to generate simulated data


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


#plot
fig2, ax2 = plt.subplots()
ax2.scatter(X,y)

### using two gaussians, one bigg in front of the other
#params of gaussian sampling
center1 = 0
std1 = 100
X = np.random.normal(center1,std1,N_SAMPLES)


#params of line
m = 3
b = 1


y = X*m + b


#params of gaussian noise
center2 = 0
std2 = 10

noise = np.random.normal(center2,std2,N_SAMPLES)

#mix line and noise
y = y + noise


#plot
fig3, ax3 = plt.subplots()
ax3.scatter(X,y)
plt.show()
