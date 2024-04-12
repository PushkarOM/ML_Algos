import numpy as np
import matplotlib.pyplot as plt
from utils_2 import *


'''Anomaly Detection
AS the name suggests we find anomalies in an given data
(after it has been filtered out of error like missing values etc).

For our problem statment we will find anomalous bhaviour in server
computer.
The dataset contains 2 feature:-
1) thorughput (mb/s)  (amount of something goind through something -- thorughput)
2) latency (ms)

While these servers are operating you collect m = 307 traning examples
of how they were behaving,and thus haave an unlabled dataset {x1,--,xm}
-- We are assuming most these examples are "normal" to find the anomalous one.



We will be using the Gaussian distibtuion as a model to find these anomalous distribution.
'''


#Loading dataset
X_train,X_val,y_val = load_data()


print("First five elemnts of X_train: ",X_train[:5])
print("First five elemnts of X_val: ",X_val[:5])
print("First five elemnts of y_val: ",y_val[:5])

print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)


plt.scatter(X_train[:, 0], X_train[:, 1],marker = 'x',c = 'b')
plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel("Latency(ms)")
plt.axis([0,30,0,30])
plt.show()

'''Gaussian Distribution
(probabilty distribution curve)
the curve is centered at mean(u) and roughly bell shaped.
Area under this curve is 1. 
mathematically the Gaussian Distribution is given by:-

p(x;u,sigma^2) = (1 / root(2*pie*(sigma^2)) ) * exp -((x-u)^2)/2*(sigma^2)
here sigma is standard deviation, and sigam^2 variance
u is mean.
for each traning data x we need to find ui and (sigma^2)i in the ith dimension.
This means:-
that each {X1,X2....Xm} X is a vector in it self, and has n featues.
So to calculate there probability:-
        p(x) = p(x1;u1,(sigma^2)1)*p(x2;u2,(sigma^2)2 )...p(xn )
    here u1 is the mean of the values inside vector x1.
    and sigma^2 is there variance.

    Now this will give an anomaly if p(x) < epsillon  #a parameter picked by us



'''

#implemeting the Gaussian Distribution or (normal , bell shaped)

def estimate_gaussian(X):
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix  --(feature vector)
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n  = X.shape

    mu = 1/m  * np.sum(X,axis = 0)
    var = 1 / m * np.sum((X-mu)**2,axis = 0)

    return mu, var


#estimating the mean and variance of each feature
mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)




#finding the value of Threshold epsillon
'''To the find the epsillon value, we will first create a cross validation dataset
and asumme we have output label for our example.
x^1cv begin the example and y^icv beign the output label 0 or 1
1 if anomalous
0 if not anomalous

For each datapoint we will calculate the probability distribution and the
we wil calculate the F1 score and find epsillion on the basis of that.


first:- we will calculate the values of truepositive(tp),falsepositive(fp),
falsenegative(fn).
tp = our pridiction 1 and output label 1
fp = our pridiction 1 and output label 0
fn = pur pridiction 0 and ouptut label 1

then we will calculate
prec = tp / (tp + fp)
rec = tp / (tp + fn)
'''

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fn = np.sum((predictions == 0) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1



p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

outliers = p_val < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
