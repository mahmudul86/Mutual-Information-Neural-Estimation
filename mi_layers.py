from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np


#this block of the code defines the structure of the neural network
#as we can see, we have two hidden layers and one output layer
class structure(layers.Layer):

  def __init__(self,hid_width, out_width,**kwargs):
    super(structure, self).__init__()
    self.hid_width=hid_width
    self.out_width=out_width
    self.dense_01 = tf.keras.layers.Dense(self.hid_width, activation=tf.nn.relu)
    self.dense_02 = tf.keras.layers.Dense(self.hid_width, activation=tf.nn.relu)
    self.dense_03 = tf.keras.layers.Dense(self.out_width, activation=None)


  def call(self, inputs):
    step_01 = self.dense_01(inputs)
    step_02 = self.dense_02(step_01)
    final_step=self.dense_03(step_02)
    return final_step
