'''Softmax Funciton'''
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('NeuralNetworksBasics\Week2\deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


'''This function is used in both softmax REgression and in 
Neural Networks when solving Multicalss classifiction.

N outputs are generated and one output is selscted as the predicted category.
Vector z is generated by a linear function which is applied to a softmax
function. Function converts this Vector Z into a prbability distributio.
After applying softmax each output will be between 0 and 1.
The funciton can be written as :-

        aj = e^zj  / sum(e^zk) where k=1 to N
    this outputs a vector of length N which consist of the probabilties.     
'''


#code implementation of softmax
def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

plt.close("all")
plt_softmax(my_softmax)

'''Few things to note about softmax:-
    * the output sum to 1
    * the softmax spans all of the outputs ie a change in
    z0 will result in change of the value of a0 - a3. (if 1 input is affected 
    it results in affecting multiple output. unlike ReLU or sigmoid in which
    one input associates to one output.)
'''


'''Cost Funciton for the Softmax'''
'''The Loss functino associated with the softmax function
,the cross entropy loss is:- -log(aj) where j ranges from 1 to n.

Now to wrtie the cost funciton we need an 'indicator function' that will be 1 when the index
matches the target and sero otherwise. as only the traget value corresponds to the loss.

            1{y == n} == {1, if y==n}
                         {0, otherwise}
So the cost function can be defined as:-
f(W,b) = -1/m[sum.m * sum.N * 1{y^j == j} log(e^zj  / sum(e^zk) where k=1 to N)] where i=1 to m and j = 1 to N
'''

#implementing the softmax function using Tensorflow


#example dataset
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

'''The obvious Organization'''
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
#as softmax is integrated into the output layer, the output is a vector of probabilties.
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))


#more accurate results can be obtained with the
'''Preferred Method'''
preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note we are using an linear activatoin here with
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- additional arguments given to this
    optimizer=tf.keras.optimizers.Adam(0.001),
)
#the from_logits=True tells the loss function that the softmax should be inculded in the lost calculation.
preferred_model.fit(
    X_train,y_train,
    epochs=10
)

#here the output is not a probability rather it can rangr from large negative value to large positive number

p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

#the output should be processed by softmax to get probabilities
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))


for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")
      


'''SparseCategoricalCrossentroy or CategoricalCrossEntropy
- SparseCategorialCrossentropy: expects the target to be an integer corresponding 
    to the index. For example, if there are 10 potential target values, y would be
    between 0 and 9. 
- CategoricalCrossEntropy: Expects the target value of an example to be one-hot 
    encoded where the value at the target index is 1 while the other N-1 entries are 
    zero. An example with 10 potential target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].
'''