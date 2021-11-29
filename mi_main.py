from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import mi_layers as mi_layers
import mi_model as mi_model
import mi_data_processing as mi_data_processing
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model


def main(args):

    input_dim=2*(args.data_dim+args.latent_dim)

    # creating the model
    def create_model():
        input=Input((input_dim,))
        struc=mi_model.MINE(args.hid_width,args.out_width,args.data_dim,args.latent_dim)
        output=struc(input)
        model = Model(input, output)
        return model

    model=create_model()

    #call the model to compute the vae loss function
    def MINE_loss(z):
        a,b,c= model(z)
        return a,b,c

    #these two lines of codes contain information about the optimization technique that we are going to use
    loss_metric = tf.keras.metrics.Mean()
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    
    #defining a single training step that includes gradient finding and parameter update
    @tf.function
    def MINE_train_step(inputs, vars):
        yt = inputs
        with tf.GradientTape() as tape:
            loss, part_01, part_02= MINE_loss(yt)
            
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))
        
        return loss, part_01, part_02
    
    #input data is loaded and reshaped here
    #using the 'mi_data_processing' file the data is reshuffled and new batches are formed before every epoch
    
    load=np.loadtxt('./io_directory/input_set.dat').astype(np.float32)    
    y=tf.reshape(load, [args.data_size,input_dim])
    frame=mi_data_processing.dataflow(y,args.data_size,args.batch_size)
    
    #here we create two vectors to accomodate the output data that we want to see
    estimated_cost_parts_vector=np.zeros((args.n_epoch,2), dtype=float)
    estimated_mutual_information_vector=np.zeros((args.n_epoch,1), dtype=float)

    #this for_loop counts for the number of epochs
    for j in range(args.n_epoch):
        print('EPOCH NUMBER:', j+1)
        #for every epoch we get a suffled dataset
        training_dataset=frame.get_shuffled_batched_dataset()
        
        #This second for loop counts for the number of iterations (i.e. number of batches) for a given epoch
        for step,train_batch in enumerate(training_dataset):
            current_vars = model.trainable_weights
            updated_loss, cost_joint, cost_marginal= MINE_train_step(train_batch, current_vars)
            #print(updated_loss,cost_joint,cost_marginal)
        
        estimated_cost_parts_vector[j,:]=[cost_joint.numpy(), cost_marginal.numpy()]
        estimated_mutual_information_vector[j,:]=[-updated_loss.numpy()]
        print('updated_mutual_information', estimated_mutual_information_vector[j,:])
        
    
    estimated_cost_parts_vector=tf.reshape(estimated_cost_parts_vector,[args.n_epoch,2])
    estimated_mutual_information_vector=tf.reshape(estimated_mutual_information_vector,[args.n_epoch,1])
    np.savetxt('./io_directory/MINE_cost_parts.dat'.format(), estimated_cost_parts_vector)
    np.savetxt('./io_directory/MINE_estimated_mutual_information.dat'.format(), estimated_mutual_information_vector)



if __name__ == '__main__':
    from configargparse import ArgParser
    p = ArgParser()
    
    # save parameters
    
    p.add('--hid_width', type=int, default=128, help='The number of neurons for the hidden layers in VAE encoder.')
    p.add('--out_width', type=int, default=1, help='The number of neurons for the hidden layers in VAE encoder.')
    p.add('--data_dim', type=int, default=10, help='The number of dimensions in the data')
    p.add('--latent_dim', type=int, default=1, help='The number of dimensions in the latent space')
    p.add('--data_size', type=int, default=20, help='The number of data points in the dataset')
    p.add("--batch_size", type=int, default=2, help='The number of data points in a batch.')
    p.add("--n_epoch", type=int, default=10, help='The number of training epochs for the program')

    #optimization hyperparams:
    p.add("--lr", type=float, default=0.001, help='Base learning rate.')
    
    args = p.parse_args()
    main(args)
