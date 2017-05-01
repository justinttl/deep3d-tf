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
            
                #removing pre-trained weights for fully connected layers so they'll be re-initialized
                del self.data_dict[u'fc6']
                del self.data_dict[u'fc7']
                del self.data_dict[u'fc8']
            
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
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        with tf.variable_scope("Pre_Processing"):
            rgb_scaled = rgb * 255.0

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
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", tracking=1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", tracking=1)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", tracking=1)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", tracking=1)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", tracking=1)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # FC Layers + Relu + Dropout
        # First Dimensions: 23040=((160//(2**5))*(288//(2**5)))*512
        self.fc6 = self.affine_layer(self.pool5, 23040, 4096, train_mode, "fc6") 
        self.fc7 = self.affine_layer(self.fc6, 4096, 4096, train_mode, "fc7")
        self.fc8 = self.affine_layer(self.fc7, 4096, 33*9*5, train_mode, "fc8")
        
        # UpScaling 
        with tf.variable_scope("FC_rs"):
            self.fc_RS = tf.reshape(self.fc8,[-1,5,9,33])
        
        scale = 16
        self.up5 = self.deconv_layer(self.fc_RS, 33, 33, scale, 0, 'up5')

        # Combine and x2 Upsample
        self.up_sum = self.up5
        
        scale = 2
        self.up = self.deconv_layer(self.up_sum, 33, 33, scale, 0, 'up', initialization='bilinear')
        self.up_conv = self.conv_layer(self.up, 33, 33, "up_conv", tracking=1)
        
        # Add + Mask + Selection
        with tf.variable_scope("mask_softmax"):
            self.mask = tf.nn.softmax(self.up_conv)
            
        self.prob  = selection.select(self.mask, rgb)

        # Clear out init dictionary
        self.data_dict = None
   
        
    # =========== Macro Layers =========== #
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, tracking=0):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, tracking)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            if tracking == 1:
                with tf.name_scope('filters'):
                    variable_summaries(filters)
                with tf.name_scope('biases'):
                    variable_summaries(biases)

            return relu
    
    def deconv_layer(self, bottom, in_channels, out_channels, scale, bias, name, initialization='default'):
        
        with tf.variable_scope(name):
            N, H, W, C = bottom.get_shape().as_list()
            shape_output = [N, scale * (H - 1) + scale * 2 - scale, scale * (W - 1) + scale * 2 - scale, out_channels] 

            filters, biases = self.get_deconv_var(2*scale, in_channels, out_channels, bias, initialization, name)
            deconv = tf.nn.conv2d_transpose(bottom, filters, shape_output, [1, scale, scale, 1])
            if bias:
                deconv = tf.nn.bias_add(deconv, biases)
            relu = tf.nn.relu(deconv)

            if tracking == 1:
                with tf.name_scope('filters'):
                    variable_summaries(filters)
                with tf.name_scope('biases'):
                    variable_summaries(biases)

            return relu
            
    def affine_layer(self, bottom, in_size, out_size, train_mode, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            relu = tf.nn.relu(fc)
            if train_mode is not None and self.trainable:
                relu = tf.nn.dropout(relu, self.dropout)
           
            if tracking == 1:
                with tf.name_scope('weights'):
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    variable_summaries(biases)


            return relu    
                
    
    # ======= Get Var Functions =========== #
    def get_conv_var(self, filter_size, in_channels, out_channels, name, tracking):

        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        del initial_value
        
        initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        del initial_value

        return filters, biases
    
    def get_deconv_var(self, filter_size, in_channels, out_channels, bias, initialization, name):

        #Initializing to bilinear interpolation
        if initialization == 'bilinear':
            C = (2 * filter_size - 1 - (filter_size % 2))/(2*filter_size)
            initial_value = np.zeros([filter_size, filter_size, in_channels, out_channels])

            for i in xrange(filter_size):
                for j in xrange(filter_size):
                    initial_value[i, j] = (1-np.abs(i/(filter_size - C))) * (1-np.abs(j/(filter_size - C)))
            initial_value = tf.convert_to_tensor(initial_value, tf.float32)
        else:
            initial_value = tf.truncated_normal([filter_size,filter_size,in_channels,out_channels],0.0,0.01)

        filters = self.get_var(initial_value, name, 0, name + "_filters")
        del initial_value

        biases = None
        if bias:
            initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
            biases = self.get_var(initial_value, name, 1, name + "_biases")
            del initial_value

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        #initialize all other weights with normal distribution with a standard deviation of 0.01
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        weights = self.get_var(initial_value, name, 0, name + "_weights")
        del initial_value

        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        del initial_value


        return weights, biases
    
    def get_bn_var(self,bottom,name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        
        #def get_var(self, initial_value, name, idx, var_name):
        gamma = self.get_var(initial_value, name, 0, name + "_gamma")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        beta = self.get_var(initial_value, name, 1, name + "_beta")
        
        return gamma, beta
    
    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
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
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
