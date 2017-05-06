import tensorflow as tf
import Deep3D_branched as deep3d
import utils
import numpy as np
import os
import os.path
import h5py

from collections import defaultdict
import pickle




def main():
    
    #importing data
    inria_file = '/a/data/deep3d_data/inria_data.h5'
    # inria_file = 'data/inria_data.h5'
    h5f = h5py.File(inria_file,'r')

    X_train_0 = h5f['X_0'][:,10:170,16:304,:]
    Y_train_0 = h5f['Y_0'][:,10:170,16:304,:]
    X_train_1 = h5f['X_1'][:,10:170,16:304,:]
    Y_train_1 = h5f['Y_1'][:,10:170,16:304,:]
    X_train_2 = h5f['X_2'][:,10:170,16:304,:]
    Y_train_2 = h5f['Y_2'][:,10:170,16:304,:]
    X_train_3 = h5f['X_3'][:,10:170,16:304,:]
    Y_train_3 = h5f['Y_3'][:,10:170,16:304,:]
    X_train_4 = h5f['X_4'][:,10:170,16:304,:]
    Y_train_4 = h5f['Y_4'][:,10:170,16:304,:]
    X_train_5 = h5f['X_5'][:,10:170,16:304,:]
    Y_train_5 = h5f['Y_5'][:,10:170,16:304,:]
    X_train_6 = h5f['X_6'][:,10:170,16:304,:]
    Y_train_6 = h5f['Y_6'][:,10:170,16:304,:]
    
    h5f.close()

    X_train = np.concatenate([X_train_0,X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6])
    Y_train = np.concatenate([Y_train_0,Y_train_1,Y_train_2,Y_train_3,Y_train_4,Y_train_5,Y_train_6])
    
    
    batchsize = 50
    num_epochs = 5
    num_batches = (X_train.shape[0]/batchsize)*num_epochs
    save_step = 10

    cost_dict = defaultdict()
    powers = np.linspace(-5,-2,8)
    learning_rates = [np.power(10,power) for power in powers]
    optimizers = ['adam','momentum']
    momentums = np.linspace(0,1,4)
    momentums = (0.98-0.14*momentums)
    count = 0
    search_count = len(optimizers) * len(learning_rates) + 2 * len(learning_rates) * (len(momentums)-1)
    for optimizer in optimizers:

        for momentum in momentums:

            for lr in learning_rates:

                print 'optimizer: ' + optimizer + ' momentum: ' + str(momentum) + ' learning rate: ' + str(lr)
                #initialize list to store outputs of run
                out_list = []

                # Define config for GPU memory debugging 
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
                config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
                config.allow_soft_placement=True
                with tf.device('/gpu:0'):
                    # Session
                    sess = tf.Session(config=config)

                    # Placeholders
                    images = tf.placeholder(tf.float32, [None, 160, 288, 3], name='input_batch')
                    true_out = tf.placeholder(tf.float32, [None, 160, 288, 3] , name='ground_truth')
                    train_mode = tf.placeholder(tf.bool, name='train_mode')

                    # Building Net based on VGG weights 
                    net = deep3d.Deep3Dnet('./vgg19.npy', dropout = 0.5)
                    net.build(images, train_mode)

                    # Define Training Objectives
                    with tf.variable_scope("Loss"):
                        #reg_factor = 1e-5
                        cost = tf.reduce_sum(tf.abs(net.prob - true_out))/batchsize


                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):

                        if optimizer == 'adam':
                            train = tf.train.AdamOptimizer(learning_rate=lr,beta1=momentum).minimize(cost)
                        if optimizer == 'momentum':
                            train = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(cost)


                    # Run initializer 
                    sess.run(tf.global_variables_initializer())

                    # Track Cost    
                    tf.summary.scalar('cost', cost)

                    # Training Loop
                    for i in xrange(num_batches):
                        # Creating Batch
                        image_mask = np.random.choice(X_train.shape[0],batchsize)
                        images_in = X_train[image_mask,:,:,:]
                        labels_in = Y_train[image_mask,:,:,:]

                        # Traing Step
                        _, cost_val = sess.run([train, cost], feed_dict={images: images_in, true_out: labels_in, train_mode: True})

                        #storing in cost_dict
                        out_list.append(cost_val)

            #saves cost outputs and clears graph for next iteration
                    cost_dict[optimizer,lr,momentum] = out_list
                    tf.reset_default_graph()

                #closing out search iteration
                count += 1
                print "finished hyperparam: " + str(count) + ' of ' + str(search_count)
                print ""
    
    # Save a cost outputs into a pickle file.
    pickle.dump(cost_dict, open( "cost_outputs.p", "wb" ) )
    return 0


if __name__ == "__main__":
    main()