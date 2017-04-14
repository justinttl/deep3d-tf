import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Deep3Dnet:
    """
    A trainable version deep3dnet.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
            
            #removing pre-trained weights for fully connected layers so they'll be re-initialized 
            del self.data_dict[u'fc6']
            del self.data_dict[u'fc7']
            del self.data_dict[u'fc8']
            
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

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        #def conv_layer(self, bottom, in_channels, out_channels, name):
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        
        
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode and self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode and self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 33*12*5, "fc8")

        self.data_dict = None
        
        
        scale = 1
        #this is is resizing the output of the fully connected layers
        self.branch5 = tf.reshape(self.fc8,[-1,33,5,12])
        
        #this section includes the branches off the main VGGnet, applying batchnorm and then a convolution
        
        #self.branch4_1 = self.batch_norm(self.pool4,train_mode,"branch4_1")
        self.branch4_1 = tf.contrib.layers.batch_norm(bottom, scale=True, is_training=phase, scope='bn')
        self.branch4_2 = self.conv_layer(self.branch4_1,512,33,"branch4_2")
        self.branch4_3 = tf.nn.relu(self.branch4_2)
        
        self.branch4_4 = self.deconv_layer(slef.branch4_3,

        """mx.symbol.Deconvolution(data=pred1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred1')"""
        scale *= 2
        
        self.branch3_1 = self.batch_norm(self.pool3,train_mode,"branch3_1")
        self.branch3_2 = self.conv_layer(self.branch4_1,512,33,"branch3_2")
        
        self.branch2_1 = self.batch_norm(self.pool3,train_mode,"branch2_1")
        self.branch2_2 = self.conv_layer(self.branch4_1,512,33,"branch2_2")
        
        self.branch1_1 = self.batch_norm(self.pool3,train_mode,"branch1_1")
        self.branch1_2 = self.conv_layer(self.branch4_1,512,33,"branch1_2")
        
        
        
        no_bias = False
        assert fuse == 'sum'
        assert method == 'multi2'
                                         
                                      
        workspace = 0
        
        
        
        pred2 = mx.symbol.Activation(data=pred2, act_type='relu')
        pred2 = mx.symbol.Deconvolution(data=pred2, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred2')
        scale *= 2
        pred3 = mx.symbol.Activation(data=pred3, act_type='relu')
        pred3 = mx.symbol.Deconvolution(data=pred3, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred3')
        scale *= 2
        pred4 = mx.symbol.Activation(data=pred4, act_type='relu')
        pred4 = mx.symbol.Deconvolution(data=pred4, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred4')
        scale *= 2
        pred5 = mx.symbol.Activation(data=pred5, act_type='relu')
        pred5 = mx.symbol.Deconvolution(data=pred5, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred5')
        feat = mx.symbol.ElementWiseSum(pred1, pred2, pred3, pred4, pred5)
        feat_act = mx.symbol.Activation(data=feat, act_type='relu', name='feat_relu')
        scale = 2
        up = mx.symbol.Deconvolution(data=feat_act, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_predup')
        up = mx.symbol.Activation(data=up, act_type='relu')
        up = mx.symbol.Convolution(data=up, kernel=(3,3), pad=(1,1), num_filter=33)
"""        
    def batch_norm(self, bottom, phase, name):        
        with tf.variable_scope(name):
            gamma, beta = self.get_bn_var(bottom,name)
            
                                         
            return tf.contrib.layers.batch_norm(bottom, center=True, scale=True, is_training=phase, scope='bn')
"""
    def get_bn_var(self,bottom,name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        
        #def get_var(self, initial_value, name, idx, var_name):
        gamma = self.get_var(initial_value, name, 0, name + "_gamma")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        beta = self.get_var(initial_value, name, 1, name + "_beta")
        
        return gamma, beta

    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu
    
    def deconv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.deconv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        
        #initialize all other weights with normal distribution with a standard deviation of 0.01
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

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

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

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
