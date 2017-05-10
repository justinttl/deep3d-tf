"""
Simple tester for the deep3d
"""
import tensorflow as tf
import Deep3D_branched as deep3d
import utils
import numpy as np
import os
import os.path
import pickle
from PIL import Image

def main():

    
    

    left_dir = "/a/data/deep3d_data/frames/train/left/"
    right_dir = "/a/data/deep3d_data/frames/train/right/"
    
    #load data into memory
    
    left_files = [left_dir + fname for fname in os.listdir(left_dir)]
    right_files = [left_dir + fname for fname in os.listdir(right_dir)]
    
    left_images = np.array([np.array(Image.open(fname)) for fname in left_files])
    right_images = np.array([np.array(Image.open(fname)) for fname in right_files]) 
    
    batchsize = 50
    num_epochs = 6
    num_batches = (left_images.shape[0]/batchsize)*num_epochs
    
    powers = np.linspace(-2.9,-2.3,4)
    learning_rates = [np.power(10,power) for power in powers]
    momentums = np.linspace(0.80,0.90,4)
    momentums = (0.99-0.14*momentums)
    count = 0
    search_count = len(learning_rates) * len(momentums)

    for momentum in momentums:

        for lr in learning_rates:

            print 'momentum: ' + str(momentum) + ' lr: ' + str(lr)
            #initialize list to store outputs of run
            out_list = []

            # Placeholders
            images = tf.placeholder(tf.float32, [None, 160, 288, 3], name='input_batch')
            true_out = tf.placeholder(tf.float32, [None, 160, 288, 3] , name='ground_truth')
            train_mode = tf.placeholder(tf.bool, name='train_mode')


            # Building Net based on VGG weights 
            net = deep3d.Deep3Dnet('./vgg19.npy', dropout = 0.5)
            net.build(images, train_mode)

            # Define Training Objectives
            with tf.variable_scope("Loss"):
                cost = tf.reduce_sum(tf.abs(net.prob - true_out))/batchsize

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):    
                train = tf.train.AdamOptimizer(learning_rate=lr,beta1=momentum).minimize(cost)
            
            
            # Define config for GPU memory debugging 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
            config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
            config.allow_soft_placement=True
            
            # Session
            sess = tf.Session(config=config)
            
            # Run initializer 
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer()) 
           
            # Track Cost    
            tf.summary.scalar('cost', cost)
            # tensorboard operations to compile summary and then write into logs
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./tensorboard_logs/'+ 'lr:' + str(lr) + 'mom:' + str(momentum) + '/', graph = sess.graph)


            # Training Loop
            print "== Start training =="

            for i in xrange(num_batches):
                image_mask = np.random.choice(left_images.shape[0],batchsize)
                images_in = left_images[image_mask,:,:,:]
                labels_in = right_images[image_mask,:,:,:]

                _, cost_val, summary = sess.run([train, cost, merged], feed_dict={images: images_in, true_out: labels_in, train_mode: True})

                writer.add_summary(summary, i)
                out_list.append(cost_val)

            count += 1
            print "finished hyperparam: " + str(count) + ' of ' + str(search_count)
            print ""
            fname = 'hyperparam_search_costs/' + 'lr' + str(lr) + '_momen' + str(momentum) + '.p'
            pickle.dump({'lr': lr, 'momentum': momentum, 'costs': out_list}, open(fname, "wb" ) )
            
            sess.close()
            tf.reset_default_graph()
                #termination stuff
            
    return 0
            
if __name__ == "__main__":
    main()