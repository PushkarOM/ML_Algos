'''Decision Boundry'''

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

#loading the data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

#plotting the current data
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

'''Now Let say we want to train a logistic regression model
on this data, and it has a form of:
    f(x) = g(w0x0 +w1x1 + b)
    where g is the sigmoid funciton
    Assuming we trained this model and we found the best value
    of b = -3, w0 = 1, w1= 1
    When we run this on our data, the resultant plot will be quit similar to the one we 
    saw earlier.
    Here is  where descision Boundry comes in, when our model 
    give and output of > 0.5 it should give an answer of 1 or yes
    and if < 0.5 it should give an output of 0 or no.
    ie :
        f_wb(x) >= 0.5 , y = 1
        f_wb(x) < 0.5 , y = 0
        this mean g(z) >= 0.5 or < 0.5
        so z >= 0 or < 0
        
    the condition can further be breaken down by putting the value of z:
        taking the current model into consideration:
        x0+x1-3 >= 0 for y=1
        x0+x1-3 < 0 for y=0

        or a more genralized sense:
        W . X + b >= 0 or < 0
'''


# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()



#plotting graph for -3+x1+x0 = 0, this is our descison boundry anything below this is 0 or no and above is 1 or yes
# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
