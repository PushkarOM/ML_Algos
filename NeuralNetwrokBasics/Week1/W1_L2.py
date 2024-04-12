#creating a simple neural netwrok to find the optimal time and temp 
#for roasting coffee
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('NeuralNetworksBasics\Week1\deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


#loading the dataset
X,Y = load_coffee_data()
print(X.shape, Y.shape)

#plotting the coffee roasting data below
plt_roast(X,Y)  #a funtion to plot data given in the module

#the red cross region denotes the sweet spot for the coffee roasting

'''Normalizing the Data'''

print("\n",f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print("\n",f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print("\n",f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print("\n",f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


#increasing the traning size of the set
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)   



'''In the "Coffee Raosting Model" there are 2 layers with sigmoid activation
    layer 0 input layer
    layer 1 hidden layer
    layer 2 output layer
'''

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)

model.summary()
#The parameter counts shown in the summary correspond to the number of elements in the weight and bias arrays as shown below.
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )


#examining the weights and baises Tensorflow has instantisted. 
'''The weights W should be of size (number of features in input, number of units in the layer)
 while the bias b size should match the number of units in the layer:
- In the first layer with 3 units, we expect W to have a size of (2,3)
 and b should have 3 elements.
- In the second layer with 1 unit, we expect W to have a size of (3,1)
 and b should have 1 element.
 '''

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("\n",f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print("\n",f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,  #defined later
)

'''above
    model.compile() defnies a loss function and 
    specifies a complie optimization
    model.fit() runs a gradient descent and fits 
    the weight to the data
'''
#printing the weights after the update
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("\n","W1:\n", W1, "\nb1:", b1)
print("\n","W2:\n", W2, "\nb2:", b2)


#loading some saved weights ##why???
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])



'''after traning this model we can use it  to make pridictions
As this a logistic model the final prediction can be categorized on the basis of
wether the value is greater that equal to 0.5
'''

X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("\n","predictions = \n", predictions)



'''Epochs and Batches

In the `compile` statement above, the number of `epochs` was set to 10. 
This specifies that the entire data set should be applied during training 10 times.
During training, you see output describing the progress of training that looks like 
this:
```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```
The first line, `Epoch 1/10`, describes which epoch the model is currently
running. For efficiency, the training data set is broken into 'batches'. 
The default size of a batch in Tensorflow is 32. There are 200000 examples 
in our expanded data set or 6250 batches. The notation on the 2nd line 
`6250/6250 [====` is describing which batch has been executed.
'''


#converting the probabilities to predictions
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print("\n",f"decisions = \n{yhat}")
#clearing the output
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")


#understanding the Layer function by plotting them
plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)
plt_output_unit(W2,b2)


#netf= lambda x : model.predict(norm_l(x))
#plt_network(X,Y,netf)