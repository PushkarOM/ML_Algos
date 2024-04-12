import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


'''Multi class-Classification
Neural Networks are used to classify data.Ex-photos of dogs
cats etc. Network of this type will have multiple units in it's
final layer.
'''

#data visulization
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
plt_mc(X_train,y_train,classes, centers, std=std)

#show classes in the dataset
print(f"Unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}") #??
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}\n")


#creating the Model

tf.random.set_seed(1234) 
model= Sequential(
    [
        Dense(2,activation='relu',name="L1"),
        Dense(4,activation='linear',name="L2")
    ]
)

#compiling and traning the network
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)


#plotting the trained models
plt_cat_mc(X_train, y_train, model, classes)
'''above the model classified the 'blobs' with an decision boundry '''

#gather the trained parameters from the first layer
l1 = model.get_layer("L1")
W1,b1 = l1.get_weights()

#plot the function of the first layer
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)

# gather the trained parameters from the output layer
l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
# create the 'new features', the training examples after L1 transformation
Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
                        x0_rng = (-0.25,np.amax(Xl2[:,0])), x1_rng = (-0.25,np.amax(Xl2[:,1])))


#explanation
'''For layer 1
We have used the activation of ReLU function (for z<0 g(z)=0,z>0 g(z)=z)
Due to the layer 1 unit 0 , the output is represented by the decison boundry(the shaded area)
the unit 0 has seprated the class 0 and 1 from 2 and 3.
point to the left of this decision boundry output 0 and right output greater than 0.
Similarly,Layer 1 unit 1 creates a decision boundry that seprates the class 0,2 from 1,3

For layer 2 (output layer)
The dots in these graph are the traning example translated by the first layer.
the output of the first layer acts as the input for this layer.
As predicted above, classes 0 and 1 have output value as 0 (a0^1), while the classes 0 and 2 have (a1^1=0)
the output as 0.(the intensity of the backgroud color indicated the highest values.)
'''