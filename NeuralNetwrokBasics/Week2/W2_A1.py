import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
plt.style.use('NeuralNetworksBasics\Week2\deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import * 

from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)


#implementing the softmax function

def my_softmax(z):
    """
        Softmax converts a vector of values to a probability distribution
    Args:
        z(ndaary (N,)) : input data, N features
    Returns:
        a (ndarry (N,)) : softmax of z
    """

    ez  = np.exp(z)
    a = ez/np.sum(ez)
    return a


z =  np.array([1.,2.,3.,4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)

print(f"My softmax:    {a}")
print(f"tensorflow softmax(z): {atf}")

'''Given is the images of handwritten digit 0 - 9. 
A Muti class classification problem'''

#load dataset
X,y = load_data()

'''the dataset contains 5000 traning examples of handwritten digits. each
of 20 x 20 pixel grayscale of the digit. the 20x20 pixel is "unrolled" into a 400 deminsional
vector.the second part of the traning set contains 5000x1 dimensional vector y, y=0 if image 
of 0, y- 4 if image of 4'''


print('The first element of x:', X[0])
print('The frist element of y:',y[0,0])
print('The frist element of y:',y[-1,0])

print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))

#visulaizing the data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

#fig.tight_layout(pad=0.5)
widgvis(fig)
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)



#implementing the model
tf.random.set_seed(1234)

model = Sequential(
    [
        tf.keras.layers.InputLayer((400,)),
        tf.keras.layers.Dense(25, activation="relu", name="L1"),
        tf.keras.layers.Dense(15, activation="relu", name="L2"),
        tf.keras.layers.Dense(10, activation="linear", name="L3")
    ], name = "my_model" 

)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()


#examining the layers
[layer1, layer2, layer3] = model.layers
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


#model with an optimizer (adam)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)


#making a prediction
image_of_two = X[1015]
display_digit(image_of_two)

prediction = model.predict(image_of_two.reshape(1,400))  # prediction

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

yhat = np.argmax(prediction_p)

print(f"np.argmax(prediction_p): {yhat}")