#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:19:20 2021

@author: thomas_yang
"""

import tensorflow as tf
import cv2
import numpy as np
from configuration import Config
import sys

def idx2class():
    return dict((v, k) for k, v in Config.pascal_voc_classes.items())

def topK(scores, K):
    B, H, W, C = scores.shape
    scores = tf.reshape(scores, shape=(B, -1))
    topk_scores, topk_inds = tf.math.top_k(input=scores, k=K, sorted=True)
    topk_clses = topk_inds % C
    topk_xs = tf.cast(topk_inds // C % W, tf.float32)
    topk_ys = tf.cast(topk_inds // C // W, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(W, tf.float32) + topk_xs, tf.int32)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_id_on_image(image, boxes):
    num_boxes = len(boxes)
    for i in range(num_boxes):
        # label class name
        class_and_score = class_and_score = "ID: {}".format(str(boxes[i, -1]))
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 2] - text_width, boxes[i, 3] - text_height), pt2=(boxes[i, 2], boxes[i, 3]), color=(0, 255, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 2] - text_width, boxes[i, 3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    return image            

def draw_boxes_ReID_on_image(image, boxes, scores, classes, tids):

    # load label index/name
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        
        # draw bbox
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
        cv2.putText(img=image, text='ID: {:d}'.format(int(tids[i])), org=(boxes[i, 0], boxes[i, 3] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
        
    return image    

def draw_boxes_joint_on_image(image, boxes, scores, classes, joint_hm=None, joint_loc=None):
    
    # find nms joint
    if type(joint_hm) != type(None):
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(joint_hm)
        keep = tf.cast(tf.equal(joint_hm, hmax), tf.float32)
        joint_hm = hmax * keep
        joint_hm = tf.image.resize(joint_hm, (image.shape[0], image.shape[1]))
        feature_shape = joint_loc.shape
        joint_loc = tf.image.resize(joint_loc, (image.shape[0], image.shape[1]))
    
    # load label index/name
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        
        # draw bbox
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
            
        # crop bbox to find max joint        
        if type(joint_hm) != type(None):
            circle_color = [(0, 0, 255), (0, 255, 0)]
            circle_radius = [5, 7]
            joint_array = (np.ones((Config.num_joints, 2)) * -1).astype(np.int)
            xmin, ymin, xmax, ymax = boxes[i]
            box_x = int((xmin + xmax)/2)
            box_y = int((ymin + ymax)/2)            
            if xmax>xmin and ymax>ymin:
                crop_hm = joint_hm[:, int(ymin+0.5):int(ymax+0.5), int(xmin+0.5):int(xmax+0.5), :]
                for j in range(5, Config.num_joints): 
                    score, _, _, ys, xs = topK(crop_hm[:, :, :, j:j+1], 1)
                    if  score[0][0].numpy() > Config.joint_threshold:
                        joint_y = int((int(ymin+0.5) + ys[0][0].numpy()))
                        joint_x = int((int(xmin+0.5) + xs[0][0].numpy()))
                        joint_array[j] = joint_x, joint_y                    
                        cv2.circle(img=image, center=(joint_x, joint_y), radius=circle_radius[j%2], color=circle_color[j%2], thickness=-1)
    
                for joint_pair in Config.skeleton:
                    star, end = joint_pair
                    if joint_array[star, 0] == -1 or joint_array[end, 0] == -1:
                        continue
                    cv2.line(img=image, pt1=(joint_array[star]),
                                        pt2=(joint_array[end]), 
                                        color=(0, 255, 255), thickness = 3)
 
                for j in range(5, Config.num_joints):  
                    x_loc = int(joint_loc[:, box_y, box_x, j*2+0]*(image.shape[1]/feature_shape[2]))
                    y_loc = int(joint_loc[:, box_y, box_x, j*2+1]*(image.shape[0]/feature_shape[1]))
                    cv2.line(img=image, pt1=(x_loc+box_x, y_loc+box_y),
                                        pt2=(box_x, box_y), 
                                        color=(255, 0, 255), thickness = 2)
                    
    return image

def draw_boxes_joint_with_location_modify_on_image(image, boxes, scores, classes, joint_hm=None, joint_loc=None):
    
    # find nms joint
    if type(joint_hm) != type(None):
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(joint_hm)
        keep = tf.cast(tf.equal(joint_hm, hmax), tf.float32)
        joint_hm = hmax * keep
        joint_hm = tf.image.resize(joint_hm, (image.shape[0], image.shape[1]))
        feature_shape = joint_loc.shape
        joint_loc = tf.image.resize(joint_loc, (image.shape[0], image.shape[1]))
    
    # load label index/name
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        
        # draw bbox
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
            
        # crop bbox to find max joint        
        if type(joint_hm) != type(None):
            circle_color = [(0, 0, 255), (0, 255, 0)]
            circle_radius = [5, 7]
            joint_array = (np.ones((Config.num_joints, 2)) * -1).astype(np.int)
            xmin, ymin, xmax, ymax = boxes[i]
            obj_c_x = int((xmin + xmax)/2)
            obj_c_y = int((ymin + ymax)/2)            
            if xmax>xmin and ymax>ymin:
                crop_hm = joint_hm[:, int(ymin+0.5):int(ymax+0.5), int(xmin+0.5):int(xmax+0.5), :]
                for j in range(5, Config.num_joints): 
                    # joint location head
                    x_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+0]*(image.shape[1]/feature_shape[2]))
                    y_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+1]*(image.shape[0]/feature_shape[1]))
                    cv2.line(img=image, pt1=(x_loc+obj_c_x, y_loc+obj_c_y),
                                        pt2=(obj_c_x, obj_c_y), 
                                        color=(255, 0, 255), thickness = 2)
                    
                    # print(crop_hm[0, :, :, j:j+1].shape)
                    weight_mask = np.zeros(crop_hm[0, :, :, j].shape)                    
                    radius = 31
                    draw_umich_gaussian(weight_mask, (x_loc+obj_c_x-xmin, y_loc+obj_c_y-ymin), radius)                    
                    crop_hm_weight = crop_hm[0, :, :, j].numpy()*weight_mask
                    crop_hm_weight = np.expand_dims(crop_hm_weight, axis = 0)
                    crop_hm_weight = np.expand_dims(crop_hm_weight, axis = 3)
                    # cv2.imshow('crop_hm_weight_before', crop_hm[0, :, :, j].numpy())
                    # cv2.imshow('crop_hm_weight_after', crop_hm_weight)
                    # cv2.imshow('weight_mask', weight_mask)
                    # if cv2.waitKey(0) == ord('q'):
                    #     sys.exit()
                    
                    _, _, _, ys, xs = topK(crop_hm_weight, 1)                    
                    ys = tf.cast(ys, tf.int32)
                    xs = tf.cast(xs, tf.int32)
                    score = crop_hm[:, ys[0][0], xs[0][0], j:j+1]
                    if  score[0][0].numpy() > Config.joint_threshold:
                        joint_y = int((int(ymin+0.5) + ys[0][0].numpy()))
                        joint_x = int((int(xmin+0.5) + xs[0][0].numpy()))
                        joint_array[j] = joint_x, joint_y                    
                        cv2.circle(img=image, center=(joint_x, joint_y), radius=circle_radius[j%2], color=circle_color[j%2], thickness=-1)
    
                for joint_pair in Config.skeleton:
                    star, end = joint_pair
                    if joint_array[star, 0] == -1 or joint_array[end, 0] == -1:
                        continue
                    cv2.line(img=image, pt1=(joint_array[star]),
                                        pt2=(joint_array[end]), 
                                        color=(0, 255, 255), thickness = 3)
 
                    
    return image

def draw_boxes_joint_with_location_modify_on_image_speedup(image, boxes, scores, classes, joint_hm=None, joint_loc=None):
    
    # find nms joint
    if type(joint_hm) != type(None):
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(joint_hm)
        keep = tf.cast(tf.equal(joint_hm, hmax), tf.float32)
        joint_hm = hmax * keep
        feature_shape = joint_loc.shape
    
    # load label index/name
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        
        # draw bbox
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
            
        # crop bbox to find max joint        
        if type(joint_hm) != type(None):
            circle_color = [(0, 0, 255), (0, 255, 0)]
            circle_radius = [5, 7]
            joint_array = (np.ones((Config.num_joints, 2)) * -1).astype(np.int)
            xmin, ymin, xmax, ymax = boxes[i] * [feature_shape[2]/image.shape[1], 
                                                 feature_shape[1]/image.shape[0], 
                                                 feature_shape[2]/image.shape[1], 
                                                 feature_shape[1]/image.shape[0]]
            obj_c_x = int((xmin + xmax)/2)
            obj_c_y = int((ymin + ymax)/2)            
            if xmax>xmin and ymax>ymin:
                crop_hm = joint_hm[:, int(ymin+0.5):int(ymax+0.5), int(xmin+0.5):int(xmax+0.5), :]  
                for j in range(5, Config.num_joints): 
                    # joint location head
                    x_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+0]*(image.shape[1]/feature_shape[2]))
                    y_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+1]*(image.shape[0]/feature_shape[1]))
                    cv2.line(img=image, pt1=(int(x_loc+obj_c_x*(Config.downsampling_ratio*image.shape[1]/Config.image_size["mobilenetv2"][1])), 
                                             int(y_loc+obj_c_y*(Config.downsampling_ratio*image.shape[0]/Config.image_size["mobilenetv2"][0]))),
                                        pt2=(int(obj_c_x*(Config.downsampling_ratio*image.shape[1]/Config.image_size["mobilenetv2"][1])), 
                                             int(obj_c_y*(Config.downsampling_ratio*image.shape[0]/Config.image_size["mobilenetv2"][0]))),
                                        color=(255, 0, 255), thickness = 2)
                    
                    # weight_mask = np.zeros(crop_hm[0, :, :, j].shape)                    
                    # radius = int(min(xmax-xmin, ymax-ymin)/2)
                    # draw_umich_gaussian(weight_mask, (x_loc*feature_shape[2]/image.shape[1]+obj_c_x-xmin, 
                                                      # y_loc*feature_shape[1]/image.shape[0]+obj_c_y-ymin), radius)                    
                    # crop_hm_weight = crop_hm[0, :, :, j].numpy()*weight_mask
                    # crop_hm_weight = np.expand_dims(crop_hm_weight, axis = 0)
                    # crop_hm_weight = np.expand_dims(crop_hm_weight, axis = 3)
                    # _, _, _, ys, xs = topK(crop_hm_weight, 1)
                    _, _, _, ys, xs = topK(crop_hm[0:1, :, :, j:j+1], 1)
                    print(ys, xs)
                    
                    ys = tf.cast(ys, tf.int32)
                    xs = tf.cast(xs, tf.int32)
                    score = crop_hm[:, ys[0][0], xs[0][0], j:j+1]
                    if  score[0][0].numpy() > Config.joint_threshold:
                        joint_x = int((int(xmin+0.5) + xs[0][0].numpy())*(Config.downsampling_ratio*image.shape[1]/Config.image_size["mobilenetv2"][1]))
                        joint_y = int((int(ymin+0.5) + ys[0][0].numpy())*(Config.downsampling_ratio*image.shape[0]/Config.image_size["mobilenetv2"][0]))                        
                        joint_array[j] = joint_x, joint_y                    
                        cv2.circle(img=image, center=(joint_x, joint_y), radius=circle_radius[j%2], color=circle_color[j%2], thickness=-1)
    
                for joint_pair in Config.skeleton:
                    star, end = joint_pair
                    if joint_array[star, 0] == -1 or joint_array[end, 0] == -1:
                        continue
                    cv2.line(img=image, pt1=(joint_array[star]),
                                        pt2=(joint_array[end]), 
                                        color=(0, 255, 255), thickness = 3)
 
                    
    return image

def draw_joint_blend_image(image, boxes, scores, classes, joint_hm, nms=True, joint_loc=None):
    
    if nms:
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(joint_hm)
        keep = tf.cast(tf.equal(joint_hm, hmax), tf.float32)
        joint_hm = hmax * keep    

    joint_hm = tf.keras.layers.UpSampling2D(interpolation='bilinear', size=(4, 4))(joint_hm).numpy()[0]
    joint_hm_max_left = np.max(joint_hm[:,:,5::2], axis = -1)
    joint_hm_max_left = (cv2.cvtColor(joint_hm_max_left, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
    joint_hm_max_left[:, :, 0::2] = 0
    joint_hm_max_right = np.max(joint_hm[:,:,6::2], axis = -1)
    joint_hm_max_right = (cv2.cvtColor(joint_hm_max_right, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
    joint_hm_max_right[:, :, 0:2] = 0
    joint_hm_max = cv2.resize(joint_hm_max_left + joint_hm_max_right, (image.shape[1], image.shape[0]))
    
    blending = cv2.addWeighted(image, 0.2, joint_hm_max, 0.8, 100)

    # load label index/name
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    feature_shape = joint_loc.shape
    joint_loc = tf.image.resize(joint_loc, (image.shape[0], image.shape[1]))
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        
        # draw bbox
        cv2.rectangle(img=blending, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=blending, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=blending, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
 
        # crop bbox to find max joint        
        if type(joint_hm) != type(None):
            xmin, ymin, xmax, ymax = boxes[i]
            obj_c_x = int((xmin + xmax)/2)
            obj_c_y = int((ymin + ymax)/2)                
            for j in range(5, Config.num_joints): 
                # joint location head
                x_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+0]*(image.shape[1]/feature_shape[2]))
                y_loc = int(joint_loc[:, obj_c_y, obj_c_x, j*2+1]*(image.shape[0]/feature_shape[1]))
                cv2.line(img=blending, pt1=(x_loc+obj_c_x, y_loc+obj_c_y),
                                       pt2=(obj_c_x, obj_c_y), 
                                       color=(255, 0, 255), thickness = 2)
    return blending