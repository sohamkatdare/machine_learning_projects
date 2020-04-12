# show correctness of Moore's Law using lineaar regression 
import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')
for line in open('moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('',r[2].split('[')[0]))
    y = int(non_decimal.sub('',r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
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
print("a:", a, "b:", b)
print("the r-squared is:", r2)

# how long does it take to double 
# log of the transistor count
# log(tc) = a * year + b
# tc = exp(b) * exp(a * year)
# 2*tc = 2 * exp(b) * exp(a * year)
#     = exp(ln(2)) * exp(b) * exp(a * year)
#     = exp(b) * exp(a * year + ln(2))
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double:", np.log(2)/a, "years")
