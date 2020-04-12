#####################################
#program to show 1D linear regression
#####################################
import numpy as np
import matplotlib.pyplot as plt

# load the data (csv files)
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
plt.scatter(X, Y)
plt.show()

#apply linear regression to get best fit line 
d = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum())/ d
b = (Y.mean()* X.dot(X) - X.mean() * X.dot(Y))/d

#predicted y values 
Y_hat = (a * X) + b

#plot calcualted regression
plt.scatter(X,Y)
plt.plot(X, Y_hat)
plt.show()


# determine how good the model is by computing the r-squared
d1 = Y - Y_hat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)