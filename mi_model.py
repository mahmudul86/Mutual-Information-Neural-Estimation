import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
import mi_layers as mi_layers

#this is the model part of the code
#the structure from 'mi_layers' called in and for the given input different parts of the cost function are calculated
class MINE(tf.keras.Model):

  def __init__(self,hid_width, out_width,data_dim,latent_dim,**kwargs):
    super(MINE, self).__init__()
    self.hid_width=hid_width
    self.out_width=out_width
    self.data_dim=data_dim
    self.latent_dim=latent_dim

    self.mine_network = mi_layers.structure(self.hid_width, self.out_width)


  def call(self, inputs):
    joint_input,marginal_input= tf.split(inputs, num_or_size_splits=2, axis=1)
    
    T_joint=self.mine_network(joint_input)
    T_marginal=self.mine_network(marginal_input)

    cost_joint_part=tf.reduce_mean(T_joint)
    cost_marginal_part=tf.math.log(tf.reduce_mean(tf.math.exp(T_marginal)))
    
    cost=-cost_joint_part+cost_marginal_part

    return cost,cost_joint_part, cost_marginal_part







