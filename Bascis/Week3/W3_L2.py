'''Logistic Regression'''
import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')


'''Sigmoid or logistic Function
For the type of data we have we need a yes or no answer,
ie our prediction will be in between 0 ans 1 or precisly 0 and 1.

This can be done by using "Sigmoid Function":

    S(x) = 1 / (1 + (e^-x))
here we use z as x (due to x already begin used as the train and test).
'''

#calculating e^z for any number
z = np.array([1,2,3])
exp_arr = np.exp(z)

print('Input to exp()',z)
print('Output of exp:',exp_arr)




'''Writing Sigmoid funciton in code'''

def sigmoid(z):
    '''
    Compute the sigmoid of z

    Args: Z (a number or array of number input)
    Returns: sigmoid of that number or numbers 
    '''

    g = 1/(1+np.exp(-z))
    return g


# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()


'''Applying this to the linear regression model

    f_wb(x^i) = g(w . (x^i)^2 + b)   === part inside bracket is z
    where g:
    g = 1/1+exp(-z)
    so,
    f_wb(x^i) = 1/1+exp(-(w . (x^i)^2 + b))

    the above eqn is Logistic Regression eqn

'''

#applying logistic regression to catogrial kind of data
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0


addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
plt.show()
plt.close('all') 