'''Logistic Regression :  Logistic Loss'''
import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')


'''In linear Regression we used sqaure error cost function
    j_wb = 1/2m (f_wb(xi)-yi)^2
    pulgging the sigmoid funciton eqaution with this give this as a 
    result.
'''


#training data
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)

#plotting the J_wb(cost function)
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()


'''As you were able to see the graph had multiple local minimum
    making it difficult for the gradient descent to find a small j_wb or 
    converger to a global minimum.(It is quit opposite to the ''Soup_bowl'')
'''
#loss:- is a measure of difference of a single example to tis target value
#Cost:- is a measure of the losses over the traning set

'''Logistic Loss Function'''

'''It is defined as:-

        loss(f_wb(xi),yi) = {
                        -log(f_wb(xi))    if yi = 1
                        -log(1-f_wb(xi))  if yi = 0
        }
        f_wb(xi) is the model prediction while yi is the target value.


    This is different from the original cost function in many
    ways. Firstly it use -log to compress the value between 0 and 1.
    It also creats 2 seprate graph 1 for when y=0 and 1 for when y=1.
'''

plt_two_logistic_loss_curves()


'''
    Now, the loss function above can be rewritten as:-
    loss(f_wb(xi),yi) = (-yi(log(f_wb(xi))))-(1-yi)log(1-f_wb(xi))

    for yi = 0, this funciton reduces to:-
        loss(f_wb(xi),0) = -log(1-f_wb(xi)) 
    
    for yi = 1, this funciton reduces to:-
        loss(f_wb(xi),1) = -log(f_wb(xi))
'''


plt.close('all')
cst = plt_logistic_cost(x_train,y_train)



