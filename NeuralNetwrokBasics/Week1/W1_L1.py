import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('NeuralNetworksBasics\Week1\deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


'''Implimenting  a simple version of neuron(creating a layer using TensorFlow)'''
'''Intro to TensorFlow and Keras
    TensorFLow is a machine learning package developed by google.
    Keras is a Framework developed independently by Francios Chollet that creats a simple,
    layer centric interface to Tensorflow.
'''



'''Neurons Without Activation Regresion/Linear Model'''

#using an example from Basics
X_train = np.array([[1.0],[2.0]],dtype=np.float32)    #size in 1000 sq feets
Y_train = np.array([[300.0],[500.0]], dtype=np.float32)

fig,ax = plt.subplots(1,1)
ax.scatter(X_train,Y_train,marker='x',c='r',label="DataPoints")
ax.legend(fontsize='xx-large')
ax.set_ylabel('Price',fontsize = 'xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()


'''The function implemented by a neuron with no activation
is the same as in course 1,linear regression
We can define a layer with one neuron or unit and compare it to the
familier linear regression function'''


linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
#there are no weights(w,b parameter)as they are not instantiated yet
linear_layer.get_weights()
#instantiaing the weights
a1 = linear_layer(X_train[0].reshape(1,1))
print("\n",a1)
w, b= linear_layer.get_weights()
print("\n",f"w = {w}, b={b}")


#setting known weights

set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print("\n",linear_layer.get_weights())



#lets compare this tensor flow model to numpy model by plotting the graph
a1 = linear_layer(X_train[0].reshape(1,1))
print("\n",a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print("\n",alin)

#making predictions on traning data using linear layers
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b


plt_linear(X_train, Y_train, prediction_tf, prediction_np)




'''Neurons With sigmoid activation'''

'''Now we will make neurons with sigmoid activation, ie using sigmoid function
to compute the output given by a neuron'''

#dataset
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

pos = Y_train == 1
neg = Y_train == 0
X_train[pos]

#plotting the data
fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()


'''Logistic Neurons
    By simply adding activation=sigmoid, to creat a sigmoid
    activated neuron.
'''

#below code explained in the later weeks
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

model.summary()
#model.summary() shows the layers and number of parameter in the model.

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print("\n",w,b)
print("\n",w.shape,b.shape)


#setting the weights to some known values
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print("\n",logistic_layer.get_weights())



#lets compare this tensor flow model to numpy model by plotting the graph
a1 = model.predict(X_train[0].reshape(1,1)) #tensor flow model
print("\n",a1)
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b) #numpy Model
print("\n",alog)


plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)