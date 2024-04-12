'''ReLU Function - Rectified linear unit'''

'''When a feature doesn't neccessarly output a binary value rather a 
range of values, ReLU funtion can be used for example, 'awarness' criteria
a particular brand, like either a person can be informed about the brand or not 
or might have. This funtion outputs values according to an on/off value. ie
    if z < 0, g(z)=0 --- where this is the on/off value.
    if z > 0, g(z)= z

    This off featuer make ReLU an Non linear activaiton
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


plt_act_trio()
_ = plt_relu_ex()
'''ReLU provieds a mean to turn off the fuction after a creatian value or before,
or whenever they are not needed'''