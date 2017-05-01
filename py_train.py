import tensorflow as tf
import Deep3D as deep3d
import utils
import numpy as np
import os
import os.path
import h5py
import matplotlib as plt

inria_file = '/a/data/deep3d_data/inria_data.h5'
h5f = h5py.File(inria_file,'r')

X_train_0 = h5f['X_0'][:,10:170,16:304,:]
Y_train_0 = h5f['Y_0'][:,10:170,16:304,:]
X_train_1 = h5f['X_1'][:,10:170,16:304,:]
Y_train_1 = h5f['Y_1'][:,10:170,16:304,:]
X_train_2 = h5f['X_2'][:,10:170,16:304,:]
Y_train_2 = h5f['Y_2'][:,10:170,16:304,:]
X_train_3 = h5f['X_3'][:,10:170,16:304,:]
Y_train_3 = h5f['Y_3'][:,10:170,16:304,:]

X_val = h5f['X_4'][:,10:170,16:304,:]
Y_val = h5f['Y_4'][:,10:170,16:304,:]
  
h5f.close()


X_train = np.concatenate([X_train_0,X_train_1,X_train_2,X_train_3])
Y_train = np.concatenate([Y_train_0,Y_train_1,Y_train_2,Y_train_3])

print "Training Size:" + str(X_train.shape)
print "Validation Size:" + str(X_val.shape)


num_batches = 100
batchsize = 32
print_step = 1
#cost_hist = []
viz_step = 10

# Define config for GPU memory debugging 
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage

with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):  
    # Session
    sess = tf.Session(config=config)
    #sess = tf.Session()
    
    # Placeholders
    images = tf.placeholder(tf.float32, [batchsize, 160, 288, 3], name='input_batch')
    true_out = tf.placeholder(tf.float32, [batchsize, 160, 288, 3] , name='ground_truth')
    train_mode = tf.placeholder(tf.bool, name='train_mode')

    # Building Net based on VGG weights 
    net = deep3d.Deep3Dnet('./vgg19.npy')
    net.build(images, train_mode)

    # Print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print 'Variable count:'
    print(net.get_var_count())
    
    # Run initializer 
    sess.run(tf.global_variables_initializer())
    
    # Define Training Objectives
    cost = tf.reduce_sum(tf.abs(net.prob - true_out), name='L1_loss')
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    
    # tensorboard operations to compile summary and then write into logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tensorboard_logs/', graph = sess.graph)

    # Training Loop
    print ""
    print "== Start training =="
    for i in xrange(num_batches):
        # Creating Batch
        image_mask = np.random.choice(X_train.shape[0],batchsize)
        images_in = X_train[image_mask,:,:,:]
        labels_in = Y_train[image_mask,:,:,:]
        
        # Traing Step
        _, cost_val, summary = sess.run([train, cost, merged], feed_dict={images: images_in, true_out: labels_in, train_mode: True})
        writer.add_summary(summary, i)

        # No longer needed: cost_hist.append(cost_val)
        if i%print_step == 0:
            print ("({}/{})".format(i, num_batches).ljust(10) + ' | Cost: ' + str(cost_val))
        
    print ""
    print "Training Completed, storing weights"
    # Store Traing Output
    net.save_npy(sess)
