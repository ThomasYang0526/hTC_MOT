#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:51:09 2021

@author: thomas_yang
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, applications
from configuration import Config
import numpy as np

   
def Res50FPN(plot_model=True):
    # inputs = tf.keras.layers.Input(shape=(Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels), name='input') 
    backbone = keras.applications.ResNet50V2(include_top=False, input_shape=(Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    c2_output, c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv2_block2_out", "conv3_block3_out", "conv4_block5_out", "conv5_block3_out"]]

    # c2_output, c3_output, c4_output, c5_output = self.backbone(images, training=training)
    p2_output = keras.layers.Conv2D(256, 1, 1, "same")(c2_output)
    p3_output = keras.layers.Conv2D(256, 1, 1, "same")(c3_output)
    p4_output = keras.layers.Conv2D(256, 1, 1, "same")(c4_output)
    p5_output = keras.layers.Conv2D(256, 1, 1, "same")(c5_output)
           
    p4_output = tf.keras.layers.add([p4_output, keras.layers.UpSampling2D()(p5_output)])
    p3_output = tf.keras.layers.add([p3_output, keras.layers.UpSampling2D()(p4_output)])
    p2_output = tf.keras.layers.add([p2_output, keras.layers.UpSampling2D()(p3_output)])

    heatmap = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='heatmap_conv1')(p2_output)
    heatmap = tf.keras.layers.BatchNormalization(name='heatmap_bn')(heatmap)
    heatmap = tf.keras.layers.ReLU(name='heatmap_relu')(heatmap)
    heatmap = tf.keras.layers.Conv2D(filters=Config.heads["heatmap"], kernel_size=(1, 1), strides=1, padding="valid", activation='sigmoid',name='heatmap_conv2')(heatmap)
    
    reg = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='reg_conv1')(p2_output)
    reg = tf.keras.layers.BatchNormalization(name='reg_bn')(reg)
    reg = tf.keras.layers.ReLU( name='reg_relu')(reg)
    reg = tf.keras.layers.Conv2D(filters=Config.heads["reg"], kernel_size=(1, 1), strides=1, padding="valid", name='reg_conv2')(reg)
    
    wh = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='wh_conv1')(p2_output)
    wh = tf.keras.layers.BatchNormalization(name='wh_bn')(wh)
    wh = tf.keras.layers.ReLU(name='wh_relu')(wh)
    wh = tf.keras.layers.Conv2D(filters=Config.heads["wh"], kernel_size=(1, 1), strides=1, padding="valid", name='wh_conv2')(wh)

    tid_embed = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='tid_conv1')(p2_output)
    tid_embed = tf.keras.layers.BatchNormalization(name='tid_bn')(tid_embed)
    tid_embed = tf.keras.layers.ReLU(name='tid_relu')(tid_embed)
    
    tid = tf.keras.layers.Conv2D(filters=Config.heads["tid"], kernel_size=(1, 1), strides=1, padding="valid", name='tid_conv2')(tid_embed)
    tid = tf.keras.layers.Softmax()(tid)

    outputs = tf.keras.layers.concatenate(inputs=[heatmap, reg, wh, tid, tid_embed], axis=-1)
    model = tf.keras.models.Model(backbone.inputs, outputs=outputs)
    model.summary()
    
    if plot_model:        
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
        print('Plot model *****************')
    return model

def Res50FPN_reID(plot_model=True):
    inputs = tf.keras.layers.Input(shape=(Config.max_boxes_per_image), name='reID')
    backbone = keras.applications.ResNet50V2(include_top=False, input_shape=(Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    c2_output, c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv2_block2_out", "conv3_block3_out", "conv4_block5_out", "conv5_block3_out"]]

    # c2_output, c3_output, c4_output, c5_output = self.backbone(images, training=training)
    p2_output = keras.layers.Conv2D(256, 1, 1, "same")(c2_output)
    p3_output = keras.layers.Conv2D(256, 1, 1, "same")(c3_output)
    p4_output = keras.layers.Conv2D(256, 1, 1, "same")(c4_output)
    p5_output = keras.layers.Conv2D(256, 1, 1, "same")(c5_output)
           
    p4_output = tf.keras.layers.add([p4_output, keras.layers.UpSampling2D()(p5_output)])
    p3_output = tf.keras.layers.add([p3_output, keras.layers.UpSampling2D()(p4_output)])
    p2_output = tf.keras.layers.add([p2_output, keras.layers.UpSampling2D()(p3_output)])

    heatmap = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='heatmap_conv1')(p2_output)
    heatmap = tf.keras.layers.BatchNormalization(name='heatmap_bn')(heatmap)
    heatmap = tf.keras.layers.ReLU(name='heatmap_relu')(heatmap)
    heatmap = tf.keras.layers.Conv2D(filters=Config.heads["heatmap"], kernel_size=(1, 1), strides=1, padding="valid", activation='sigmoid',name='heatmap_conv2')(heatmap)
    
    reg = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='reg_conv1')(p2_output)
    reg = tf.keras.layers.BatchNormalization(name='reg_bn')(reg)
    reg = tf.keras.layers.ReLU( name='reg_relu')(reg)
    reg = tf.keras.layers.Conv2D(filters=Config.heads["reg"], kernel_size=(1, 1), strides=1, padding="valid", name='reg_conv2')(reg)
    
    wh = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='wh_conv1')(p2_output)
    wh = tf.keras.layers.BatchNormalization(name='wh_bn')(wh)
    wh = tf.keras.layers.ReLU(name='wh_relu')(wh)
    wh = tf.keras.layers.Conv2D(filters=Config.heads["wh"], kernel_size=(1, 1), strides=1, padding="valid", name='wh_conv2')(wh)

    tid_embed = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='tid_conv1')(p2_output)
    tid_embed = tf.keras.layers.BatchNormalization(name='tid_bn')(tid_embed)
    tid_embed = tf.keras.layers.ReLU(name='tid_relu')(tid_embed)
    
    idx = tf.cast(inputs, dtype=tf.int32)
    # print(tid_embed.shape)
    tid = tf.keras.layers.Reshape(target_shape=(tid_embed.shape[1]*tid_embed.shape[2], tid_embed.shape[-1]))(tid_embed)
    # print(tid.shape)
    tid = tf.gather(params=tid, indices=idx, batch_dims=1)
    # print(tid.shape)
    tid = tf.expand_dims(tid, axis=2)
    # print(tid.shape)    
    tid = tf.keras.layers.Conv2D(filters=Config.heads["tid"], kernel_size=(1, 1), strides=1, padding="valid", name='tid_conv2')(tid)
    # print(tid.shape)
    tid = tf.keras.layers.Softmax()(tid)
    
    outputs = tf.keras.layers.concatenate(inputs=[heatmap, reg, wh, tid_embed], axis=-1)
    model = tf.keras.models.Model([backbone.inputs, inputs], outputs=[outputs, tid])
    # model.summary()
    
    if plot_model:        
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
        print('Plot model *****************')
    return model

if __name__ == '__main__':
    model = Res50FPN_reID()
    pre = model([np.ones((Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels)), 
                 np.ones((Config.batch_size, Config.max_boxes_per_image, 2))], True)

    