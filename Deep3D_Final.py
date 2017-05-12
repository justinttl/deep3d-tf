import tensorflow as tf

import numpy as np
from functools import reduce
import os.path
import selection

VGG_MEAN = [103.939, 116.779, 123.68]


class Deep3Dnet:
    """
    A trainable version deep3dnet.
    """

    def __init__(self, deep3d_path=None, trainable=True, dropout=0.5):
        if deep3d_path is not None:
            if os.path.isfile(deep3d_path):
                self.data_dict = np.load(deep3d_path, encoding='latin1').item()
                
                if deep3d_path.endswith('vgg19.npy'):
                    print "Using VGG19 Weights, Reinitializing fully connected layers"
                    # removing pre-trained weights for fully connected layers 
                    # so they'll be re-initialized
                    del self.data_dict[u'fc6']
                    del self.data_dict[u'fc7']
                    del self.data_dict[u'fc8']
                else:
                    print "Weights given, initializing based on stored params shown:"
            
            else:
                self.data_dict = None
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on,
                                                                 and batchnorm will behave differently
        """
        with tf.variable_scope("Pre_Processing"):
            
            rgb_scaled = rgb / 255.0
            
            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [160, 288, 1]
            assert green.get_shape().as_list()[1:] == [160, 288, 1]
            assert blue.get_shape().as_list()[1:] == [160, 288, 1]
            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            assert bgr.get_shape().as_list()[1:] == [160, 288, 3]

        # Convolution Stages
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1",                train_mode, trainable=1)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2",      train_mode, tracking=1,trainable=1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1",       train_mode,trainable=1)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2",    train_mode, tracking=1,trainable=1)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1",      train_mode,trainable=1)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2",    train_mode,trainable=1)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3",    train_mode,trainable=1)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4",    train_mode, tracking=1,trainable=1)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1",      train_mode,trainable=1)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2",    train_mode,trainable=1)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3",    train_mode,trainable=1)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4",    train_mode, tracking=1,trainable=1)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1",      train_mode, trainable=1)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2",    train_mode, trainable=1)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3",    train_mode, trainable=1)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4",    train_mode, tracking=1,trainable=1)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        # FC Layers + Relu + Dropout
        self.fc6 = self.affine_layer(self.pool5, 23040, 4096, "fc6",         train_mode, tracking=1) 
        self.fc7 = self.affine_layer(self.fc6, 4096, 4096, "fc7",            train_mode, tracking=1)
        self.fc8 = self.affine_layer(self.fc7, 4096, 33*9*5, "fc8",          train_mode, tracking=1, dropout=1.0)
        
        # Upscaling last branch
        with tf.variable_scope("FC_rs"):
            self.fc_RS = tf.reshape(self.fc8,[-1,5,9,33])


        #-------branch 1-----
        scale = 1
        self.bn_pool1 = self.batch_norm(self.pool1, "branch1_bn", train_mode)
        self.branch1_1 = self.conv_layer(self.bn_pool1, 64, 33, "branch1_conv", train_mode)
        self.branch1_2 = self.deconv_layer(self.branch1_1, 33, 33, scale, 0, 'branch1_upconv', train_mode, tracking=1)

        #-------branch 2-----
        scale *= 2
        self.bn_pool2 = self.batch_norm(self.pool2, "branch2_bn", train_mode)
        self.branch2_1 = self.conv_layer(self.bn_pool2, 128, 33, "branch2_conv", train_mode)
        self.branch2_2 = self.deconv_layer(self.branch2_1, 33, 33, scale, 0, 'branch2_upconv', train_mode, tracking=1)

        # -------branch 3-----
        scale *= 2
        self.bn_pool3 = self.batch_norm(self.pool3, "branch3_bn", train_mode)
        self.branch3_1 = self.conv_layer(self.bn_pool3, 256, 33, "branch3_conv", train_mode)
        self.branch3_2 = self.deconv_layer(self.branch3_1, 33, 33, scale, 0, 'branch3_upconv', train_mode, tracking=1)

        # -------branch 4-----
        scale *= 2
        self.bn_pool4 = self.batch_norm(self.pool4, "branch4_bn", train_mode)
        self.branch4_1 = self.conv_layer(self.bn_pool4, 512, 33, "branch4_conv", train_mode)
        self.branch4_2 = self.deconv_layer(self.branch4_1, 33, 33, scale, 0, 'branch4_upconv', train_mode, tracking=1)
        
        # -------branch 5-----
        scale *= 2
        self.branch5_1 = tf.nn.relu(self.fc_RS)
        self.branch5_2 = self.deconv_layer(self.branch5_1, 33, 33, scale, 0, 'branch5_upconv', train_mode, tracking=1, relu=0)

        # Combine and x2 Upsample
        self.up_sum = self.branch1_2 + self.branch2_2 + self.branch3_2 + self.branch4_2 + self.branch5_2
        scale = 2
        self.up = self.deconv_layer(self.up_sum, 33, 33, scale, 0, 'up', train_mode, tracking=1)
        
        # Last Conv Layer
        self.up_conv = self.conv_layer(self.up, 33, 33, "up_conv", train_mode, tracking=1)
        
        # Tracking presoftmax activation
        with tf.name_scope('up_conv_act'):
            variable_summaries(self.up_conv)
        
        # Add + Mask + Selection
        with tf.variable_scope("mask_softmax"):
            self.mask = tf.nn.softmax(self.up_conv)
            self.mask.set_shape([None, 160,288, 33])

        # Tracking Mask  
        with tf.name_scope('mask_act'):
            variable_summaries(self.mask)
        
        self.prob  = selection.select(self.mask, rgb)

        # Clear out init dictionary
        self.data_dict = None
   
        
    # =========== Macro Layers =========== #
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def batch_norm(self, bottom, name, train_mode, trainable=1):
        with tf.variable_scope(name):
            
            mean, variance = tf.nn.moments(bottom, [0,1,2], keep_dims=False)   
            
            decay = 0.6
            prev_mean, prev_var, offset, scale = self.get_bn_var(bottom, train_mode, name, trainable)
            new_mean = (1-decay)*mean + decay*prev_mean
            new_var = (1-decay)*variance + decay*prev_var
            
            ema_mean = tf.cond(train_mode, lambda: new_mean, lambda: prev_mean)
            ema_var = tf.cond(train_mode, lambda: new_var, lambda: prev_var)
            
            assign_pmean_op = prev_mean.assign(ema_mean)
            assign_pvar_op = prev_var.assign(ema_var)
            
            with tf.control_dependencies([assign_pmean_op]):
                curr_mean = tf.identity(tf.cond(train_mode, lambda: mean, lambda: prev_mean))
            with tf.control_dependencies([assign_pvar_op]):
                curr_var = tf.identity(tf.cond(train_mode, lambda: variance, lambda: prev_var))         
          
            return  tf.nn.batch_normalization(bottom, curr_mean, curr_var, offset, scale, 1e-6, name=name)
        
        
    def conv_layer(self, bottom, in_channels, out_channels, name,
                   train_mode, batchnorm=0, tracking=0, trainable=1):

        with tf.variable_scope(name):
            filters, biases = self.get_conv_var(3, in_channels, out_channels, name, trainable)
            conv = tf.nn.conv2d(bottom, filters, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)

            relu = tf.nn.relu(bias)

            if tracking == 1:
                with tf.name_scope('filters'):
                    variable_summaries(filters)
                with tf.name_scope('biases'):
                    variable_summaries(biases)
                    
            return relu
    

    def deconv_layer(self, bottom, in_channels, out_channels, 
                     scale, bias, name,
                     train_mode, initialization='default', batchnorm=0, tracking = 0, trainable=1, relu=1):
        
        with tf.variable_scope(name):
            #N, H, W, C = bottom.get_shape().as_list()
            
            dyn_input_shape = tf.shape(bottom)
            N = dyn_input_shape[0]
            H = dyn_input_shape[1]
            W = dyn_input_shape[2]
            C = dyn_input_shape[3]

            shape_output = tf.stack([N, 
                                     scale * (H - 1) + scale * 2 - scale,
                                     scale * (W - 1) + scale * 2 - scale,
                                     out_channels])
            
            filters, biases = self.get_deconv_var(2*scale, in_channels, out_channels, bias, initialization, name, trainable)
            deconv = tf.nn.conv2d_transpose(bottom, filters, shape_output, [1, scale, scale, 1])

            if bias: 
                deconv = tf.nn.bias_add(deconv, biases)

            if relu:
                deconv = tf.nn.relu(deconv)

            if tracking == 1:
                with tf.name_scope('filters'):
                    variable_summaries(filters)
                if bias:
                    with tf.name_scope('biases'):
                        variable_summaries(biases)

            return deconv

    def affine_layer(self, bottom, in_size, out_size, name,
                     train_mode, batchnorm=0, tracking=0, trainable=1, dropout=0.5):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        
            relu = tf.nn.relu(fc)
            
            relu_do = tf.nn.dropout(relu, self.dropout)
                
            if tracking == 1:
                with tf.name_scope('weights'):
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    variable_summaries(biases)
           
            return tf.cond(train_mode, lambda: relu_do, lambda: relu)
        
        
    # ======= Get Var Functions =========== #
    def get_bn_var(self, bottom, train_mode, name, trainable):                             
        N, H, W, C = bottom.get_shape().as_list()
        
        initial_value = tf.ones([C])
        scale = self.get_var(initial_value, name, 0, name + "_scale", trainable)
        
        initial_value = tf.zeros([C])
        offset = self.get_var(initial_value, name, 1, name + "_offset", trainable)
        
        initial_value = tf.zeros([C])
        estimated_mean = self.get_var(initial_value, name, 2, name + "_estmean", trainable)
        
        initial_value = tf.ones([C])
        estimated_var = self.get_var(initial_value, name, 3, name + "_estvar", trainable)
  
        return estimated_mean, estimated_var, offset, scale

    
    def get_conv_var(self, filter_size, in_channels, out_channels,
                     name , trainable):

        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)
        
        initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)
        

        return filters, biases
    
    def get_deconv_var(self, filter_size, in_channels, out_channels, 
                       bias, initialization,
                       name, trainable):

        #Initializing to bilinear interpolation
        if initialization == 'bilinear':
            C = (filter_size - 1 - ((filter_size/2) % 2))/(filter_size)
            initial_value = np.zeros([filter_size, filter_size, in_channels, out_channels])
            for i in xrange(filter_size):
                for j in xrange(filter_size):
                    initial_value[i, j] = (1-np.abs(i/(filter_size/2.0 - C))) * (1-np.abs(j/(filter_size/2.0 - C)))
            initial_value = tf.convert_to_tensor(initial_value, tf.float32)
        else:
            initial_value = tf.truncated_normal([filter_size,filter_size,in_channels,out_channels],0.0,0.01)


        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)
        
        biases = None
        if bias:
            initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
            biases = self.get_var(initial_value, name, 1, name + "_biases")

        #del initial_value
        return filters, biases

    def get_fc_var(self, in_size, out_size, 
                   name, trainable):
        #initialize all other weights with normal distribution with a standard deviation of 0.01
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)
        #del initial_value

        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)
        #del initial_value


        return weights, biases
 
    
    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name, trainable=trainable)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()
       
        return var
    
    
    # =========== Util Functions ========= # 
    def save_npy(self, sess, npy_path="./deep3d-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        
            
        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path
    
    
    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
    