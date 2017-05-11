import tensorflow as tf
import Deep3D_Final as deep3d
import utils
import numpy as np
import os
import os.path
from collections import defaultdict
import pickle
import sys


def train(in_path, out_path):
    print "-- Training Deep3D --"

    # Calculate number of iterations 
    batchsize = 50
    num_epochs = 5
    left_dir = "/a/data/deep3d_data/frames2/left/"
    right_dir = "/a/data/deep3d_data/frames2/right/"
    iter_per_epoch = (len(os.listdir(left_dir))/batchsize)
    num_iter = iter_per_epoch * num_epochs
    print "Number of Training Iterations : " + str(num_iter)

    validation_size = 2000
    
    # Learning Rates 
    lr = 0.0012
    b1 = 0.84
    b2 = 0.99
    #lr_decay = 0.5
    
    # Define CPU operations for filename queue and random shuffle batch queue
    with tf.device('/cpu:0'):
        left_image_queue = tf.train.string_input_producer(
          left_dir + tf.convert_to_tensor(os.listdir(left_dir)[:-validation_size:]),
          shuffle=False, num_epochs=num_epochs)
        right_image_queue = tf.train.string_input_producer(
          right_dir + tf.convert_to_tensor(os.listdir(right_dir)[:-validation_size:]),
          shuffle=False, num_epochs=num_epochs)

        # use reader to read file
        image_reader = tf.WholeFileReader()

        _, left_image_raw = image_reader.read(left_image_queue)
        left_image = tf.image.decode_jpeg(left_image_raw)
        left_image = tf.cast(left_image, tf.float32)/255.0

        _, right_image_raw = image_reader.read(right_image_queue)
        right_image = tf.image.decode_jpeg(right_image_raw)
        right_image = tf.cast(right_image, tf.float32)/255.0

        left_image.set_shape([160,288,3])
        right_image.set_shape([160,288,3])

        # preprocess image
        batch = tf.train.shuffle_batch([left_image, right_image], 
                                       batch_size = batchsize,
                                       capacity = 12*batchsize,
                                       num_threads = 1,
                                       min_after_dequeue = 4*batchsize)
        

    # ------ GPU Operations ---------- #
    # Define config for GPU memory debugging 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
    config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
    config.allow_soft_placement=True

    # Session
    sess = tf.Session(config=config)

    # Placeholders
    images = tf.placeholder(tf.float32, [None, 160, 288, 3], name='input_batch')
    true_out = tf.placeholder(tf.float32, [None, 160, 288, 3] , name='ground_truth')
    train_mode = tf.placeholder(tf.bool, name='train_mode')

    # Building Net based on VGG weights 
    net = deep3d.Deep3Dnet(in_path, dropout = 0.5)
    net.build(images, train_mode)

    # Print number of variables used
    print 'Variable count:'
    print(net.get_var_count())

    # Define Training Objectives
    with tf.variable_scope("Loss"):
        cost = tf.reduce_sum(tf.abs(net.prob - true_out))/batchsize

    train = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2 =b2).minimize(cost)

    # Run initializer 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) 
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Track Cost    
    tf.summary.scalar('cost', cost)
    # Tensorboard operations to compile summary and then write into logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./final_trainlog/', graph = sess.graph)


    # ---------- Training Loop --------------- #
    print ""
    print ">> Start training <<"
    
    print_step = 1
    save_step = 25
    print "saving every " + str(save_step) + " iterations"
    
    # Base case data fetch
    next_batch = sess.run(batch)
    count = 0
    try:
        while not coord.should_stop():
            # Traing Step
            _, cost_val, next_batch, summary,up_conv = sess.run([train, cost, batch, merged,net.up_conv], 
                                                                feed_dict={images: next_batch[0],
                                                                           true_out: next_batch[1],
                                                                           train_mode: True})
            writer.add_summary(summary, count)
            if count%print_step == 0:
                print str(count).ljust(10) + ' | Cost: ' + str(cost_val).ljust(10) + ' | UpConv Max: ' + str(np.mean(up_conv[0], axis =(0,1)).max())
            if count%save_step == 0:
                print "Checkpoint Save"
                net.save_npy(sess, npy_path = out_path)
            count = count + 1     

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Store Final Traing Output
    print ""
    print "Training Completed, storing weights"
    net.save_npy(sess, npy_path = out_path)
                                                                                                       
    # Terminate session                                                    
    coord.join(queue_threads)
    sess.close()
                    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Invalid Arguements. Give 1) input weights file 2) output weight file"
    else:
        train(sys.argv[1], sys.argv[2])