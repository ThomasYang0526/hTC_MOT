#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:01:57 2021

@author: thomas_yang
"""

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from DCNv2 import DCNv2  
from configuration import Config

class BasicBlock():
    def __init__(self, planes, name, stride = 1, dilation = 1):
        self.channels = planes
        self.stride = stride
        self.dilation = (dilation, dilation)
        self.name = name
    def __call__(self, x, residual=None):
        if residual is None:
            residual = x
        padding = 'same'
        if self.stride == 2:
            x = KL.ZeroPadding2D(((1,1),(1,1)))(x)
            padding = 'valid'
        x = KL.Conv2D(self.channels, 3, name = self.name + '.conv1', strides = self.stride, padding = padding, use_bias = False, dilation_rate = self.dilation)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn1')(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(self.channels, 3, name = self.name + '.conv2', strides = 1, padding = 'same', use_bias = False, dilation_rate = self.dilation)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn2')(x)
        x = KL.Add()([x, residual])
        x = KL.Activation('relu')(x)
        return x
    
class Root():
    def __init__(self, out_channels, residual, name):
        self.channels = out_channels
        self.residual = residual
        self.name = name
    def __call__(self, x):
        children = x
        x = KL.Concatenate(axis = -1)(x)
        x = KL.Conv2D(self.channels, 1, name = self.name + '.conv', strides = 1, use_bias = False, padding='same')(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn')(x)
        if self.residual:
            x = KL.Add()([x, children[0]])
        x = KL.Activation('relu')(x)
        return x
        

class Tree():
    def __init__(self, levels, out_channels, name, stride = 1, level_root = False, root_dim = 0, root_kernel_size = 1, dilation = 1, root_residual = False):
        if root_dim == 0:
            self.root_dim = 2 * out_channels
        else:
            self.root_dim = root_dim
        self.level_root = level_root
        self.levels = levels
        self.root_kernel_size = root_kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.root_residual = root_residual
        self.name = name
        
    def tree1(self, x, residual, children, level, out_channels, stride = 1, dilation = 1, root_kernel_size=1, root_residual=False):
        if level == 1:
            x = BasicBlock(out_channels, self.name + '.tree1', stride, dilation = dilation)(x, residual)
        else:
            x = Tree(level-1, out_channels, self.name + '.tree1', stride, root_dim=0, root_kernel_size=root_kernel_size, dilation = dilation, root_residual = root_residual)(x, residual, children)
        return x
    
    def tree2(self, x, residual, children, level, out_channels, dilation = 1, root_kernel_size=1, root_residual=False):
        if level == 1:
            x = BasicBlock(out_channels, self.name + '.tree2', 1, dilation = dilation)(x, residual)
        else:
            x = Tree(level-1, out_channels, self.name + '.tree2', 1, root_dim=self.root_dim+out_channels, root_kernel_size=root_kernel_size, dilation = dilation, root_residual = root_residual)(x, residual, children)
        return x
            
    def downsample(self, x):
        if self.stride > 1:
            x = KL.MaxPooling2D(self.stride, self.stride)(x)
        return x
    def project(self, x):
        if self.in_channels != self.out_channels:
            x = KL.Conv2D(self.out_channels, 1, name = self.name + '.project.0', strides = 1, use_bias = False)(x)
            x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.project.1')(x)
        return x
    
    def root(self, x):
        return Root(self.out_channels, self.root_residual, self.name + '.root')(x)
    
    def __call__(self, x, residual = None, children=None):
        self.in_channels = x.get_shape()[-1]
        if self.level_root: self.root_dim += self.in_channels
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual, None, self.levels, self.out_channels, self.stride, self.dilation, self.root_kernel_size, self.root_residual)
        if self.levels == 1:
            x2 = self.tree2(x1, None, None, self.levels, self.out_channels, self.dilation, self.root_kernel_size, self.root_residual)
            x = self.root([x2, x1, *children])
        else:
            children.append(x1)
            x = self.tree2(x1, None, children, self.levels, self.out_channels, self.dilation, self.root_kernel_size, self.root_residual)
        return x
    

def make_conv_level(x, num_filters, level, name, stride = 1, dilation = 1):
    for i in range(level):
        padding = 'same'
        if stride == 2:
            x = KL.ZeroPadding2D(((1,1),(1,1)))(x)
            padding = 'valid'
        x = KL.Conv2D(num_filters, 3, name = name + '.0', strides = stride if i == 0 else 1, \
                      padding = padding, \
                      use_bias = False,\
                      dilation_rate = (dilation, dilation))(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = name + '.1')(x)
        x = KL.Activation('relu')(x)
    return x
        

class Base():
    def __init__(self, levels, channels, num_classes = 1000, residual_root = False):
        self.channels = channels
        self.residual_root = residual_root
        self.levels = levels
        self.num_classes = num_classes
        self.name = 'base'
        
    def __call__(self, x):
        y = []
        #base layer
        x = KL.Conv2D(self.channels[0], 7, name = self.name + '.base_layer.0', padding = 'same', use_bias = False)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.base_layer.1')(x)
        x = KL.Activation('relu')(x)
        #level0
        x = make_conv_level(x, self.channels[0], self.levels[0], name = self.name + '.level0')
        y.append(x)
        #level1
        x = make_conv_level(x, self.channels[1], self.levels[1], name = self.name + '.level1', stride = 2)
        y.append(x)
        #level2
        for i in range(2, 6):
            x = Tree(self.levels[i], self.channels[i], name = self.name + '.level'+str(i), stride = 2, level_root = i > 2, root_residual = self.residual_root)(x)
            y.append(x)
        return y

class IDAUp():
    def __init__(self, o, channels, up_f, name):
        self.channels = channels
        self.o = o
        self.up_f = up_f
        self.name = name
    
    def DepthwiseConv2DTranspose(self, x, kernel_size, name = None, pad = 1, strides = 1, use_bias = False):
        #keras and tensorflow grouped transpose convolutinoal unsupported
        # print('x', x.shape)
        up_x = KL.Lambda(lambda x : tf.reshape(tf.transpose(tf.reshape(tf.concat([x, tf.tile(tf.zeros_like(x), [1, 1, 1, strides*strides-1])], axis = -1), [tf.shape(x)[0], x.shape[1], x.shape[2], strides, strides, x.shape[3]]), [0, 1, 3, 2, 4, 5]), [tf.shape(x)[0], x.shape[1]*strides, x.shape[2]*strides, x.shape[3]]))(x)
        up_x = KL.ZeroPadding2D(((pad, pad - (strides - 1)), (pad, pad - (strides - 1))))(up_x)
        # print('up_x', up_x.shape)
        return  KL.DepthwiseConv2D((kernel_size, kernel_size), name = name, use_bias = use_bias)(up_x)

    def __call__(self, x, startp, endp):
        for i in range(startp + 1, endp):
            # x[i] = DCNv2(self.o, 3, name = self.name + '.proj_%d.conv'%(i-startp))(x[i])
            x[i] = KL.Conv2D(self.o, 3, name = self.name + '.proj_%d.conv'%(i-startp), strides = 1, padding = 'same', use_bias = False)(x[i])
            x[i] = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.proj_%d.actf.0'%(i-startp))(x[i])
            x[i] = KL.Activation('relu')(x[i])
            
            
            x[i] = self.DepthwiseConv2DTranspose(x[i], kernel_size = self.up_f[i-startp]*2, \
                                                  name = self.name + '.up_%d'%(i-startp), \
                                                  pad = self.up_f[i-startp] * 2 - 1 - self.up_f[i-startp]//2, \
                                                  strides = self.up_f[i-startp])             
                
            # print(x)
            x[i] = KL.Add()([x[i], x[i-1]])
            # x[i] = DCNv2(self.o, 3, name = self.name + '.node_%d.conv'%(i-startp))(x[i])
            x[i] = KL.Conv2D(self.o, 3, name = self.name + '.node_%d.conv'%(i-startp), strides = 1, padding = 'same', use_bias = False)(x[i])
            x[i] = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.node_%d.actf.0'%(i-startp))(x[i])
            x[i] = KL.Activation('relu')(x[i])
            
import numpy as np    
class DLAUp():
    def __init__(self, startp, channels, scales):
        self.startp = startp
        self.in_channels = channels
        self.channels = list(channels)
        self.scales = np.array(scales, dtype = int)
        self.name = 'dla_up'
        
    def __call__(self, x, ):
        out = [x[-1]]
        for i in range(len(x) - self.startp - 1):
            j = -i - 2
            IDAUp(self.channels[j], self.in_channels[j:], self.scales[j:]//self.scales[j], name = self.name + '.ida_%d'%i)(x, len(x)-i-2, len(x))
            out.insert(0, x[-1])
            self.scales[j+1:] = self.scales[j]
            self.in_channels[j+1:] = [self.channels[j] for _ in self.channels[j+1:]]
        return out
            
class DLASeg():
    def __init__(self, heads, down_ratio, final_kernel, last_level, head_conv, out_channel=0):
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.heads = heads
        self.head_conv = head_conv
        self.out_channel = out_channel
        self.final_kernel = final_kernel
        self.down_ratio = down_ratio
        self.name = 'dlaseg'
        
    def detection(self, hm, wh, ids, reg, num_classes = 1, K = 128):
        hm = tf.nn.sigmoid(hm)
        hmax = tf.nn.max_pool2d(hm, 3, 1, 'SAME')
        km = tf.where(tf.equal(hmax, hm), hmax, tf.zeros_like(hmax))
        bs, h, w, c = km.shape
        bs = tf.shape(km)[0]   
        
        scores, indices = tf.nn.top_k(tf.reshape(km, [bs, -1]), K)
        
        classes = indices // (h * w)
        y, x = (indices % (h * w)) // w, (indices % (h * w)) % w
        batch_index = tf.reshape(tf.range(bs), [bs, 1]) * tf.ones_like(classes) 
        index = tf.stack([batch_index, y, x, classes], axis = -1)
        kwh = tf.gather_nd(tf.reshape(wh, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        kid = tf.gather_nd(tf.reshape(ids, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        krg = tf.gather_nd(tf.reshape(reg, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        
        kid = tf.nn.l2_normalize(kid, axis = -1)
        x, y = tf.reshape(tf.cast(x, 'float32'), [-1, 1]) + krg[:, 0:1], tf.reshape(tf.cast(y, 'float32'), [-1, 1]) + krg[:, 1:2]
        
        bboxes = tf.concat([x - kwh[:, 0:1]/2, \
                            y - kwh[:, 1:2]/2, \
                            x + kwh[:, 0:1]/2, \
                            y + kwh[:, 1:2]/2, \
                            tf.reshape(scores, [-1, 1]), \
                            tf.reshape(tf.cast(classes, 'float32'), [-1, 1]), \
                            tf.reshape(kid, [-1, tf.shape(kid)[-1]])], axis = -1)
        bboxes = tf.reshape(bboxes, [bs, -1, 4 + 2 + tf.shape(kid)[-1]])
        #[bs, number of boxes, boxes + class + score + feature] = [bs, N, 4 + 2 + 512] = [bs, N, 518]
        return bboxes
        
        
        
    def __call__(self,x):
        _base = Base([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512])
        # _base = Base([1, 1, 1, 2, 2, 1], [8, 16, 32, 64, 128, 256])
        x = _base(x)
        channels = _base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        x = DLAUp(self.first_level, channels[self.first_level:], scales)(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
            
        if self.out_channel == 0:
            out_channel = channels[self.first_level]
        IDAUp(out_channel, channels[self.first_level:self.last_level], [2 ** i for i in range(self.last_level - self.first_level)], self.name + '.idaup')(y, 0, len(y))
        outputs = []
        for head in self.heads:
            classes = self.heads[head]
            if self.head_conv > 0:
                x = KL.Conv2D(self.head_conv, 3, name = self.name +'.' + head + '.conv1', padding = 'same', activation='relu')(y[-1])
                if head == 'heatmap':
                    x = KL.Conv2D(classes, self.final_kernel, name = self.name + '.' + head + '.conv2', padding = 'same', activation='sigmoid')(x)
                else:
                    x = KL.Conv2D(classes, self.final_kernel, name = self.name + '.' + head + '.conv2', padding = 'same')(x)
            else:
                x = KL.Conv2D(classes, self.final_kernel, name = self.name + '.' + head + '.conv', padding = 'same')(y[-1])
            outputs.append(x)
        
        detection = KL.concatenate(outputs, axis=-1)
        # detection = KL.Lambda(lambda x : self.detection(*x))(outputs)
        #[bs, number of boxes, boxes + class + score + feature] = [bs, N, 4 + 2 + 512] = [bs, N, 518]
        return detection
        
def DLA_MODEL():
    # inputs = KL.Input(shape = [608, 1088, 3])
    inputs = KL.Input(shape = [Config.get_image_size()[0], Config.get_image_size()[1], 3])
    # outputs = DLASeg(heads = {'hm': 1, 'wh': 2, 'id': 128, 'reg': 2},\
    outputs = DLASeg(heads = Config.heads,\
                     down_ratio = 4,\
                     final_kernel = 1,\
                     last_level = 5,\
                     head_conv=256
                     )(inputs)

    model = KM.Model(inputs, outputs)
    # model.summary()    
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    print('Plot model *****************')
    return model

if __name__ == '__main__':
    model = DLA_MODEL()
    pre = model(np.ones((Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels)), True)
    # tf.keras.models.save_model(model, filepath='/home/thomas_yang/ML/CenterNet_TensorFlow2/saved_model/' + "saved_model", include_optimizer=False, save_format="tf")
    # model.save_weights(filepath='/home/thomas_yang/ML/CenterNet_TensorFlow2/saved_model/'+"test-model.h5", save_format="h5")



