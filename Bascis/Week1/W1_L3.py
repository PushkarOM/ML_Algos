'''Implemeting the gradient descent'''


import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients


'''
Sample data for this lab will be
size   price
1      300
2      500
.in relative units
'''

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

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


'''Gradient Descent
    Gradient descent is basically derivative of cost function
    wrt W and once with B to compute the values of W anf B for which 
    cost function has low value.
    i.e 

    w = w - (alpha)(d of j_wb wrt w)   d of j_wb = dj_dw
    b = b - (alpha)(d of j_wb wrt B)

    where alpha is the learning rate, anf j_wb is the standered cost function

    Now, lets start to implement the Gradient descent
'''


def compute_gradient(x, y, w, b):
    '''
    Args:
        x_train,y_train, w&b parameters
    Returns:
        dj_dw
        dj_db
    '''

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i])* x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train,y_train, compute_cost, compute_gradient)
#the above line help in plotting the data the code for computing the cost and gradient is the same here and lab_utils
plt.show()

'''The above plot shows the value of dj_dw for a fix b=100
    i.e the slope of the graph at that point.
    as the graphs shape is like a bowl, the slop on the left
    is -ve and on the right is +ve. i.e on constant updating 
    of w this will lead to minimum value of w

'''

#finally the Gradient Descent


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    '''
    Args:
        x_train,y_train
        w&b's initial value
        aplha: learning rate
        num of iteration
        cost_function:     function to call to produce cost
        gradient_function: function to call to produce gradient
    
    Reutrns:
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (list): History of parameters [w,b] 
    history meaning how these parametes change over number of 
    iterations
    '''    
    w = copy.deepcopy(w_in) #helps avoiding changes in the global prameter
    #an array to store J and P_history
    J_history = []
    P_history = []
    b = b_in
    w = w_in



    for i in range(num_iters):

        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)


        #updating the parametes
        b = b - alpha*dj_db
        w = w - alpha*dj_dw

        #save cost J at each iteration
        if i<100000:
            J_history.append(cost_function(x,y,w,b))
            P_history.append([w,b])

         # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, P_history #return w and J,w history for graphing


#triggering the gradient descent
w_init = 0
b_init = 0

iterations =  10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")



'''the above gradient converges to a local minimum as observed in the graph

    The cost starts large and rapidly declines as described in the slide from the lecture.
    - The partial derivatives, `dj_dw`, and `dj_db` also get smaller, rapidly at first and then more slowly. As shown in the diagram from the lecture, as the process nears the 'bottom of the bowl' progress is slower due to the smaller value of the derivative at that point.
    - progress slows though the learning rate, alpha, remains fixed

'''


#cost ve iteration plot
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()



#plotting the Progress of gradient descent
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt.show()

'''
    Learning Rate alpha
    this decides how long of a jump
    i.e how much each parameter changes every time gradient descent is 
    ran. adqaute of aplha will result in a working model like above, but
    if the value is too high gradient descent might overshoot global minimum
    and never find it, or if the value is too small it might take forever for larger dataset to 
    reach global minimum

    below the case of high alpha is visualized
'''


#let's change the alpha parameter
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)

plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
