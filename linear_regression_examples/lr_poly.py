import numpy as np 
import matplotlib.pyplot as plt 

X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

#convert to numpy array 
X = np.array(X)
Y = np.array(Y)

#plot to see what data looks like 
plt.scatter(X[:,1], Y)
plt.show()


w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# to plot our quadratic model predictions, let's
# create a line of x's and calculate the predicted y's
x_line = np.linspace(X[:,1].min(), X[:,1].max())
y_line = w[0] + w[1] * x_line + w[2] * x_line * x_line
plt.plot(x_line, y_line)
plt.show()


# determine how good the model is by computing the r-squared
#Yhat = X.dot(w)
d1 = Y - Y_hat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)