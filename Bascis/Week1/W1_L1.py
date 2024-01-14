'''Implementing MODEL FUNCTION'''
import numpy as np
import matplotlib.pyplot as plt
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

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]   #this can also be done using len()
print(f"Number of traning examples is: {m}")

#Traning examples are denoted by x_i and y_i or x^i and y^i
i = 0 #incerease this value to see other traning examples

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i})),(x^({i}))=({x_i},{y_i})")

#plotting the data (Data visulaization)
'''plt.scatter(x_train,y_train,marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel("Price in (100s of doller)")
plt.xlabel("SIze (1000 sqft)")
plt.show()
''' #<---remove this to see

#creating the model Function
'''
f_wb(x^i) = w*x^i+b

the value f_wb -- is y-hat or the predicted value of our system
currently this function is just a straight line passin through origin
setting the values of w and b will decides it's inclination.
'''

#setting values for w and b
w = 200
b = 100

#function to compute the y-hat or model function

def compute_model_output(x, w, b):
    '''
    This function calculates the values of f_wb 
    for every traning example ie i=0 to i=m-1 (1 to m in actula counting)
    Args:
        x_train, w & b prameters 
    '''

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    
    return f_wb

#calling the function to calculate the y-hat
tmp_f_wb = compute_model_output(x_train, w,b)

#plotting the model's prediction 
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

#plotting the data points
plt.scatter(x_train, y_train,marker='x',c='r',label='Actual values')
plt.ylabel("Price in (100s of doller)")
plt.xlabel("SIze (1000 sqft)")
#plt.show()


#changing the values of w and b above to try and fit it in the points.

#now that the model is complete let's compute a new value's prediction let x = 1.2 (in 1000's sq ft)
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} Thousand Dollars")



'''
THis is how you implement an model function in ML
'''