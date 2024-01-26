'''Multiple linear regression'''

import numpy as np
import matplotlib.pyplot as plt
import copy,math


'''
    Problem Statement
We need to predict house pricing on the basis of 4 parameters
size, bedrooms, floors,age of the house

'''

#loading the data
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#as these are numpy arrays
print(f"X_train.shape: {X_train.shape}, X type:{type(X_train)}")
print(X_train)
print(f"y_train.shape: {y_train.shape}, y type:{type(y_train)}")
print(y_train)

'''Here W parameter will be a vector with N elements
    each element contains the parameter associated with one
    feature, so as we have 4 feature we have 4 value of w for each one.
    b remains a scaler
'''

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


'''New Model eqaution
    As now we have multiple feature our equations becomes:

     f_wb = w0x0 + w1x1 + w2x2 ..... +w4x4 + b
i.e. f_wb = w . x + b

'''

#one way of calculating this is induvidually calculate the value of wi*xi then add the bias parameter (b)

def predict_single_loop(X,w,b):

    m = X.shape[0]
    p = 0
    for i in range(m):
        p_i = w[i]*X[i]
        p += p_i

    p += b
    return p


#p above is our f_wb and p_i calculates the value of wi*xi element

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#f_wb shape () signifies that f_wb is a scaler


'''We can do what we did above with vector 
calculation by taking dot of w and x vector'''

def predict(X,w,b):

    p = np.dot(w,X) + b
    return p

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#using this makes calculations much faster, so we will use this going forward


'''Computing the Cost function with multiple variable'''

def compute_cost(X,y,w,b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i =  np.dot(X[i],w)+b  
        cost += (f_wb_i - y[i])**2
    cost /= (2*m)

    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


'''computing gradient and gradient descent with multiple varible'''

def compute_gradient(X,y,w,b):
    
    m,n = X.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


'''

    above mainly 2 lines have changed,
    err = (np.dot(X[i], w) + b) - y[i] === here we directly calculate the value of 
    cost, X[i] denotes the first row of the featuers ie x1,x2,x3,x4 in row 1 which is 
    dotted with w1,w2,w3,w4.

    
'''


def gradient_descent(X,y,w_int,b_int,cost_function,gradient_function,alpha,iteration):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_int)  #avoid modifying global w within function
    b = b_int
    
    for i in range(iteration):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(iteration / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")




# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()