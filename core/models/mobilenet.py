#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:41:20 2021

@author: thomas_yang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:41:20 2021

@author: thomas_yang
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from configuration import Config
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Add, DepthwiseConv2D, UpSampling2D

#%%

def _dilate_conv_block(inputs, filters, kernel, dilation_rate, training=None, name=None):    
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=(1, 1), dilation_rate=dilation_rate, use_bias=False, name=name+'conv2D')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'bn')(x)
    x = tf.keras.layers.ReLU(name=name+'relu')(x)
    return x

def _conv_block(inputs, filters, kernel, strides, training=None, name=None):    
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides, use_bias=False, name=name+'conv2D')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'bn')(x)
    x = tf.keras.layers.ReLU(name=name+'relu')(x)
    return x

def _conv_block_without_relu(inputs, filters, kernel, strides, training=None, name=None):    
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides, use_bias=False, name=name+'conv2D')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'bn')(x)
    return x

def _bottleneck(inputs, filters, kernel, t, s, r=False, training=None, name=None):
    
    tchannel = inputs.shape[-1] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1), training=training, name=name+'conv_block_')

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same', use_bias=False, name=name+'dw')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn1')(x)
    x = tf.keras.layers.ReLU(name=name+'relu')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=name+'conv2d')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn2')(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n, training=None, name=None):
    x = _bottleneck(inputs, filters, kernel, t, strides, training=training, name = name + 'bottleneck0_')

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True, training=training, name = name + 'bottleneck'+str(i)+'_')

    return x

def _make_transposed_conv_layer(inputs, num_filters, num_kernels, FPN, training=None, name=None):
    x = tf.keras.layers.Conv2DTranspose(filters=num_filters[0], kernel_size=num_kernels[0], strides=2, padding="same", use_bias=False, name=name+'tpcv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'bn1')(x)
    x = tf.keras.layers.add([x, FPN[2]])
    x = tf.keras.layers.ReLU(name=name+'relu1')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=num_filters[1], kernel_size=num_kernels[1], strides=2, padding="same", use_bias=False, name=name+'tpcv2')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn2')(x)
    x = tf.keras.layers.add([x, FPN[1]])
    x = tf.keras.layers.ReLU(name=name+'relu2')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=num_filters[2], kernel_size=num_kernels[2], strides=2, padding="same", use_bias=False, name=name+'tpcv3')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn3')(x)
    x = tf.keras.layers.add([x, FPN[0]])
    x = tf.keras.layers.ReLU(name=name+'relu3')(x)
    return x

def _make_upsample2d_layer(inputs, num_filters, num_kernels, FPN, training=None, name=None):
    x = tf.compat.v1.image.resize(inputs, (inputs.shape[1]*2, inputs.shape[2]*2))
    x = _conv_block_without_relu(x, num_filters[0], kernel=(3, 3), strides=(1, 1), training=training, name=name + 'conv2_1_')
    x = tf.keras.layers.add([x, FPN[2]])
    x = tf.keras.layers.ReLU(name=name+'relu1')(x)
    x = tf.compat.v1.image.resize(x, (x.shape[1]*2, x.shape[2]*2))
    x = _conv_block_without_relu(x, num_filters[0], kernel=(3, 3), strides=(1, 1), training=training, name=name + 'conv2_2_')
    x = tf.keras.layers.add([x, FPN[1]])
    x = tf.keras.layers.ReLU(name=name+'relu2')(x)
    x = tf.compat.v1.image.resize(x, (x.shape[1]*2, x.shape[2]*2))
    x = _conv_block_without_relu(x, num_filters[0], kernel=(3, 3), strides=(1, 1), training=training, name=name + 'conv2_3_')
    x = tf.keras.layers.add([x, FPN[0]])
    x = tf.keras.layers.ReLU(name=name+'relu3')(x)
    return x


def _downsample_conv_layer(inputs, num_filters, kernel, strides, training=None, name=None):
    x = tf.keras.layers.Conv2D(filters=num_filters[0], kernel_size=kernel, strides=strides, padding='same', use_bias=False, name=name+'conv2D1')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name+'bn1')(x)
    x = tf.keras.layers.ReLU(name=name+'relu1')(x)
    x = tf.keras.layers.Conv2D(filters=num_filters[1], kernel_size=kernel, strides=strides, padding='same', use_bias=False, name=name+'conv2D2')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn2')(x)
    x = tf.keras.layers.ReLU(name=name+'relu2')(x)
    x = tf.keras.layers.Conv2D(filters=num_filters[2], kernel_size=kernel, strides=strides, padding='same', use_bias=False, name=name+'conv2D3')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'bn3')(x)
    x = tf.keras.layers.ReLU(name=name+'relu3')(x)
    return x

def MobileNetV2(training=None, plot_model=True):
    times = 1  
    inputs = tf.keras.layers.Input(shape=(Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels), name='input')    
    # backbone
    x = _conv_block(inputs, 32*times, kernel=(3, 3), strides=(2, 2), training=training, name='conv1_')
    x = _inverted_residual_block(x, 16*times, (3, 3), t=1, strides=1, n=1, training=training, name='invert1_')
    x = _inverted_residual_block(x, 24*times, (3, 3), t=6, strides=2, n=2, training=training, name='invert2_')
    s4 = x
    x = _inverted_residual_block(x, 32*times, (3, 3), t=6, strides=2, n=3, training=training, name='invert3_')
    s8 = x
    x = _inverted_residual_block(x, 64*times, (3, 3), t=6, strides=2, n=4, training=training, name='invert4_')
    x = _inverted_residual_block(x, 96*times, (3, 3), t=6, strides=1, n=3, training=training, name='invert5_')
    s16 = x
    x = _inverted_residual_block(x, 160*times, (3, 3), t=6, strides=2, n=3, training=training, name='invert6_')
    x = _inverted_residual_block(x, 320*times, (3, 3), t=6, strides=1, n=1, training=training, name='invert7_')
       
    # neck
    neck_channel = 128
    s4 = tf.keras.layers.Conv2D(filters=neck_channel, kernel_size=(1, 1), strides=1, padding="valid", name='s4')(s4)
    s4 = tf.keras.layers.BatchNormalization(name='s4_bn1')(s4)
    s8 = tf.keras.layers.Conv2D(filters=neck_channel, kernel_size=(1, 1), strides=1, padding="valid", name='s8')(s8)
    s8 = tf.keras.layers.BatchNormalization(name='s8_bn1')(s8)    
    s16 = tf.keras.layers.Conv2D(filters=neck_channel, kernel_size=(1, 1), strides=1, padding="valid", name='s16')(s16)
    s16 = tf.keras.layers.BatchNormalization(name='s16_bn1')(s16) 
    feature_map = _conv_block(x, neck_channel, kernel=(3, 3), strides=(1, 1), training=training, name='conv2_')
    num_filters = [neck_channel, neck_channel, neck_channel]
    num_kernels = [4, 4, 4]
    FPN = [s4, s8, s16]
    
    up_num = 0 
    stage_num_bbox = 1
    stage_num_joint = 3
    kernel_neck = (3, 3)
    dilation_rate = (2, 2)
    
    bbox_up = feature_map    
    for i in range(up_num+1):
        bbox_up = _make_transposed_conv_layer(bbox_up, num_filters, num_kernels, FPN, training=training, name='bbox_transPoseConv%d_' %i)
        if i == up_num:
            break
        bbox_up = _downsample_conv_layer(bbox_up, num_filters, kernel=(3, 3), strides=(2, 2), training=training, name='bbox_downSample%d_' %i)
    
    bbox_stage = bbox_up
    for i in range(stage_num_bbox+1):
        bbox_stage_ = _conv_block(bbox_stage, neck_channel, kernel=kernel_neck, strides=(1, 1), training=training, name='neck_bbox3x3_1_%d_' %i)
        bbox_stage_ = _conv_block(bbox_stage_, neck_channel, kernel=kernel_neck, strides=(1, 1), training=training, name='neck_bbox3x3_2_%d_' %i)
        bbox_stage_ = _dilate_conv_block(bbox_stage_, neck_channel, kernel=kernel_neck, dilation_rate=dilation_rate, training=training, name='neck_dilate_bbox3x3_1_%d_' %i)
        bbox_stage_ = _dilate_conv_block(bbox_stage_, neck_channel, kernel=kernel_neck, dilation_rate=dilation_rate, training=training, name='neck_dilate_bbox3x3_2_%d_' %i)
        bbox_stage_ = _conv_block(bbox_stage_, neck_channel, kernel=(1, 1), strides=(1, 1), training=training, name='neck_bbox1x1%d_' %i)
        if i == stage_num_bbox:
            break
        if i == 0:
            bbox_stage = tf.keras.layers.add([bbox_stage_, bbox_up])
        else:
            bbox_stage = tf.keras.layers.add([bbox_stage, bbox_stage_, bbox_up])

    # joint_up = feature_map
    # for i in range(up_num+1):
    #     joint_up = _make_transposed_conv_layer(joint_up, num_filters, num_kernels, FPN, training=training, name='joint_transPoseConv%d_' %i)
    #     if i == up_num:
    #         break
    #     joint_up = _downsample_conv_layer(joint_up, num_filters, kernel=(3, 3), strides=(2, 2), training=training, name='joint_downSample%d_' %i) 
    
    # joint_stage = joint_up
    # for i in range(stage_num_joint+1):
    #     joint_stage_ = _conv_block(joint_stage, neck_channel, kernel=kernel_neck, strides=(1, 1), training=training, name='neck_joint3x3_1_%d_' %i)
    #     joint_stage_ = _conv_block(joint_stage_, neck_channel, kernel=kernel_neck, strides=(1, 1), training=training, name='neck_joint3x3_2_%d_' %i)
    #     joint_stage_ = _dilate_conv_block(joint_stage_, neck_channel, kernel=kernel_neck, dilation_rate=dilation_rate, training=training, name='neck_dilate_joint3x3_1_%d_' %i)
    #     joint_stage_ = _dilate_conv_block(joint_stage_, neck_channel, kernel=kernel_neck, dilation_rate=dilation_rate, training=training, name='neck_dilate_joint3x3_2_%d_' %i)
    #     joint_stage_ = _conv_block(joint_stage_, neck_channel, kernel=(1, 1), strides=(1, 1), training=training, name='neck_joint1x1%d_' %i)
    #     if i == stage_num_joint:
    #         break
    #     if i == 0:
    #         joint_stage = tf.keras.layers.add([joint_stage_, joint_up])
    #     else:
    #         joint_stage = tf.keras.layers.add([joint_stage, joint_stage_, joint_up])
    
    # detect head 
    heatmap = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='heatmap_conv1')(bbox_stage)
    heatmap = tf.keras.layers.BatchNormalization(name='heatmap_bn')(heatmap)
    heatmap = tf.keras.layers.ReLU(name='heatmap_relu')(heatmap)
    heatmap = tf.keras.layers.Conv2D(filters=Config.heads["heatmap"], kernel_size=(1, 1), strides=1, padding="valid", activation='sigmoid',name='heatmap_conv2')(heatmap)
    
    reg = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='reg_conv1')(bbox_stage)
    reg = tf.keras.layers.BatchNormalization(name='reg_bn')(reg)
    reg = tf.keras.layers.ReLU( name='reg_relu')(reg)
    reg = tf.keras.layers.Conv2D(filters=Config.heads["reg"], kernel_size=(1, 1), strides=1, padding="valid", name='reg_conv2')(reg)
    
    wh = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='wh_conv1')(bbox_stage)
    wh = tf.keras.layers.BatchNormalization(name='wh_bn')(wh)
    wh = tf.keras.layers.ReLU(name='wh_relu')(wh)
    wh = tf.keras.layers.Conv2D(filters=Config.heads["wh"], kernel_size=(1, 1), strides=1, padding="valid", name='wh_conv2')(wh)
    
    #joint = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='joint_conv1')(joint_stage)
    #joint = tf.keras.layers.BatchNormalization(name='joint_bn')(joint)
    #joint = tf.keras.layers.ReLU(name='joint_relu')(joint)
    #joint = tf.keras.layers.Conv2D(filters=Config.heads["joint"], kernel_size=(1, 1), strides=1, padding="valid", activation='sigmoid', name='joint_conv2')(joint)

    #joint_loc = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='joint_loc_conv1')(joint_stage)
    #joint_loc = tf.keras.layers.BatchNormalization(name='joint_loc_bn')(joint_loc)
    #joint_loc = tf.keras.layers.ReLU(name='joint_loc_relu')(joint_loc)
    #joint_loc = tf.keras.layers.Conv2D(filters=Config.heads["joint_loc"], kernel_size=(1, 1), strides=1, padding="valid", activation=None, name='joint_loc_conv2')(joint_loc)
    
    
    # outputs=[heatmap, reg, wh, joint, joint_loc]
    # outputs=[heatmap, reg, wh]
    outputs = tf.keras.layers.concatenate(inputs=[heatmap, reg, wh], axis=-1)
    model = tf.keras.models.Model(inputs, outputs=outputs)
    model.summary()
    if plot_model:
        print('Plot model *****************')
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model



if __name__ == '__main__':
    model = MobileNetV2()
    pre = model(np.ones((Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels)), True)
    # tf.keras.models.save_model(model, filepath='/home/thomas_yang/ML/CenterNet_TensorFlow2/saved_model/' + "saved_model", include_optimizer=False, save_format="tf")
    # model.save_weights(filepath='/home/thomas_yang/ML/CenterNet_TensorFlow2/saved_model/'+"test-model.h5", save_format="h5")






