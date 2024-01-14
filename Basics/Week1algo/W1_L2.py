'''Implementing the COST FUNCTION'''

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

'''
Sample data for this lab will be
size   price
1      300
2      500
.in relative units
'''
x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])


'''Computing the cost function

    J_wb = 1/2m * {sumassion from i=0 to i=m-1}(f_wb(x^i) - y^i)^2
    where f_wb is comput function
    f_wb is out prediction and y^i is the actual value.

'''

#defning the compute_cost function
def compute_cost(x,y,w,b):
    '''
    Args:
        x_train,y_train, w&b parameters
    
    Reutrns:
        Total_cost : computes cost value
    '''

    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
    
    total_cost = (1/(2*m))*cost_sum

    return total_cost


'''What is this costs function?/
    Your goal in developing any alogirithms is that 
    it accuratly predicts the value for a given input x.
    here being HOuse price based on the value of size of the
    house.

    The Cost function the measure of how accurate the model is
    on the traning data

    Now lets visulaize the cost function by fixing the value of b = 100
    and focus on w
'''

plt_intuition(x_train,y_train)



#plotting a 3D contoure plot to understand how w and b change

#using a larger dataset
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()

