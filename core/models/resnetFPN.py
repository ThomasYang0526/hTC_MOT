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

# def get_backbone_ResNet50(input_shape):
#     """Builds ResNet50 with pre-trained imagenet weights"""
#     backbone = keras.applications.ResNet50(include_top=False, input_shape=input_shape)
#     c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
#     return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

# def get_backbone_ResNet50V2():
#     """Builds ResNet50 with pre-trained imagenet weights"""
#     backbone = keras.applications.ResNet50V2(include_top=False, input_shape=(Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
#     c2_output, c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
#     return keras.Model(inputs=[backbone.inputs], outputs=[c2_output, c3_output, c4_output, c5_output])

# def get_backbone_ResNet101(input_shape):
#     """Builds ResNet101 with pre-trained imagenet weights"""
#     backbone = keras.applications.ResNet101(include_top=False, input_shape=input_shape)
#     c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]]
#     return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

# class customFeaturePyramid2(keras.models.Model):
#     """Builds the Feature Pyramid with the feature maps from the backbone.
#     Attributes:
#       num_classes: Number of classes in the dataset.
#       backbone: The backbone to build the feature pyramid from.
#         Currently supports ResNet50, ResNet101 and V1 counterparts.
#     """

#     def __init__(self, backbone=None, **kwargs):
#         super(customFeaturePyramid2, self).__init__(name="customFeaturePyramid2", **kwargs)
#         self.backbone = backbone if backbone else get_backbone_ResNet50V2()
#         self.conv_c2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
#         self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
#         self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
#         self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
#         self.upsample_2x = keras.layers.UpSampling2D()

#     def call(self, images, training=False):
#         c2_output, c3_output, c4_output, c5_output = self.backbone(images, training=training)
#         p2_output = self.conv_c2_1x1(c2_output)
#         p3_output = self.conv_c3_1x1(c3_output)
#         p4_output = self.conv_c4_1x1(c4_output)
#         p5_output = self.conv_c5_1x1(c5_output)
#         p4_output = p4_output + self.upsample_2x(p5_output)
#         p3_output = p3_output + self.upsample_2x(p4_output)
#         p2_output = p2_output + self.upsample_2x(p3_output)

#         heatmap = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='heatmap_conv1')(p2_output)
#         heatmap = tf.keras.layers.BatchNormalization(name='heatmap_bn')(heatmap)
#         heatmap = tf.keras.layers.ReLU(name='heatmap_relu')(heatmap)
#         heatmap = tf.keras.layers.Conv2D(filters=Config.heads["heatmap"], kernel_size=(1, 1), strides=1, padding="valid", activation='sigmoid',name='heatmap_conv2')(heatmap)
        
#         reg = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='reg_conv1')(p2_output)
#         reg = tf.keras.layers.BatchNormalization(name='reg_bn')(reg)
#         reg = tf.keras.layers.ReLU( name='reg_relu')(reg)
#         reg = tf.keras.layers.Conv2D(filters=Config.heads["reg"], kernel_size=(1, 1), strides=1, padding="valid", name='reg_conv2')(reg)
        
#         wh = tf.keras.layers.Conv2D(filters=Config.head_conv["mobilenetv2"], kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name='wh_conv1')(p2_output)
#         wh = tf.keras.layers.BatchNormalization(name='wh_bn')(wh)
#         wh = tf.keras.layers.ReLU(name='wh_relu')(wh)
#         wh = tf.keras.layers.Conv2D(filters=Config.heads["wh"], kernel_size=(1, 1), strides=1, padding="valid", name='wh_conv2')(wh)

#         outputs = tf.keras.layers.concatenate(inputs=[heatmap, reg, wh], axis=-1)
#         # model = tf.keras.models.Model(inputs, outputs=outputs)
#         # model.summary()
#         # if plot_model:
#             # print('Plot model *****************')
#             # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
#         return outputs
    
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

    outputs = tf.keras.layers.concatenate(inputs=[heatmap, reg, wh], axis=-1)
    model = tf.keras.models.Model(backbone.inputs, outputs=outputs)
    model.summary()
    
    if plot_model:
        print('Plot model *****************')
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model

if __name__ == '__main__':
    model = Res50FPN()
    pre = model(np.ones((Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels)), True)

    