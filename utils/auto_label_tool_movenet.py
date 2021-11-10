#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:33:39 2021

@author: thomas_yang
"""

import tensorflow as tf
# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import os
import numpy as np

from configuration import Config
from core.centernet import CenterNet, PostProcessing
from data.dataloader import DataLoader
from core.models.mobilenet import MobileNetV2
from core.centernet import Decoder

import tensorflow_hub as hub
# from tensorflow_docs.vis import embed

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import HTML, display

#%%
# Dictionary to map joints of body part
KEYPOINT_DICT = {
     'nose':0,
     'left_eye':1,
     'right_eye':2,
     'left_ear':3,
     'right_ear':4,
     'left_shoulder':5,
     'right_shoulder':6,
     'left_elbow':7,
     'right_elbow':8,
     'left_wrist':9,
     'right_wrist':10,
     'left_hip':11,
     'right_hip':12,
     'left_knee':13,
     'right_knee':14,
     'left_ankle':15,
     'right_ankle':16
 } 
 
 # map bones to matplotlib color name
KEYPOINT_EDGE_INDS_TO_COLOR = {
     (0,1): 'm',
     (0,2): 'c',
     (1,3): 'm',
     (2,4): 'c',
     (0,5): 'm',
     (0,6): 'c',
     (5,7): 'm',
     (7,9): 'm',
     (6,8): 'c',
     (8,10): 'c',
     (5,6): 'y',
     (5,11): 'm',
     (6,12): 'c',
     (11,12): 'y',
     (11,13): 'm',
     (13,15): 'm',
     (12,14): 'c',
     (14,16): 'c'
 } 

def draw_prediction_on_image(
     image, keypoints_with_scores, crop_region=None, close_figure=False,
     output_image_height=None):
   """Draws the keypoint predictions on image"""
   height, width, channel = image.shape
   aspect_ratio = float(width) / height
   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
   # To remove the huge white borders
   fig.tight_layout(pad=0)
   ax.margins(0)
   ax.set_yticklabels([])
   ax.set_xticklabels([])
   plt.axis('off')
   im = ax.imshow(image)
   line_segments = LineCollection([], linewidths=(4), linestyle='solid')
   ax.add_collection(line_segments)
   # Turn off tick labels
   scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
   (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)
   line_segments.set_segments(keypoint_edges)
   line_segments.set_color(edge_colors)
   if keypoint_edges.shape[0]:
     line_segments.set_segments(keypoint_edges)
     line_segments.set_color(edge_colors)
   if keypoint_locs.shape[0]:
     scat.set_offsets(keypoint_locs)
   if crop_region is not None:
     xmin = max(crop_region['x_min'] * width, 0.0)
     ymin = max(crop_region['y_min'] * height, 0.0)
     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
     rect = patches.Rectangle(
         (xmin,ymin),rec_width,rec_height,
         linewidth=1,edgecolor='b',facecolor='none')
     ax.add_patch(rect)
   fig.canvas.draw()
   image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
   image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
   plt.close(fig)
   if output_image_height is not None:
     output_image_width = int(output_image_height / height * width)
     image_from_plot = cv2.resize(
         image_from_plot, dsize=(output_image_width, output_image_height),
          interpolation=cv2.INTER_CUBIC)
   return image_from_plot

def _keypoints_and_edges_for_display(keypoints_with_score,height,
                                      width,keypoint_threshold=0.11):
   """Returns high confidence keypoints and edges"""
   keypoints_all = []
   keypoint_edges_all = []
   edge_colors = []
   num_instances,_,_,_ = keypoints_with_score.shape
   for id in range(num_instances):
     kpts_x = keypoints_with_score[0,id,:,1]
     kpts_y = keypoints_with_score[0,id,:,0]
     kpts_scores = keypoints_with_score[0,id,:,2]
     kpts_abs_xy = np.stack(
         [width*np.array(kpts_x),height*np.array(kpts_y)],axis=-1)
     kpts_above_thrs_abs = kpts_abs_xy[kpts_scores > keypoint_threshold,: ]
     keypoints_all.append(kpts_above_thrs_abs)
     for edge_pair,color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
       if (kpts_scores[edge_pair[0]] > keypoint_threshold and 
           kpts_scores[edge_pair[1]] > keypoint_threshold):
         x_start = kpts_abs_xy[edge_pair[0],0]
         y_start = kpts_abs_xy[edge_pair[0],1]
         x_end = kpts_abs_xy[edge_pair[1],0]
         y_end = kpts_abs_xy[edge_pair[1],1]
         lien_seg = np.array([[x_start,y_start],[x_end,y_end]])
         keypoint_edges_all.append(lien_seg)
         edge_colors.append(color)
   if keypoints_all:
     keypoints_xy = np.concatenate(keypoints_all,axis=0)
   else:
     keypoints_xy = np.zeros((0,17,2))
   if keypoint_edges_all:
     edges_xy = np.stack(keypoint_edges_all,axis=0)
   else:
     edges_xy = np.zeros((0,2,2))
   return keypoints_xy,edges_xy,edge_colors 

#%%
# model_name = "movenet_lightning"
model_name = "movenet_thunder_f16.tflite"

if "tflite" in model_name:
  if "movenet_lightning_f16" in model_name:
    !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
    input_size = 192
  elif "movenet_thunder_f16" in model_name:
    !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite
    input_size = 256
  elif "movenet_lightning_int8" in model_name:
    !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite
    input_size = 192
  elif "movenet_thunder_int8" in model_name:
    !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
  elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoint_with_scores = outputs['output_0'].numpy()
    return keypoint_with_scores
#%%

def idx2class():
    return dict((v, k) for k, v in Config.pascal_voc_classes.items())


def draw_boxes_on_image(image, boxes, scores, classes, joint_hm=None):
    
    # find nms joint
    if type(joint_hm) != type(None):
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(joint_hm)
        keep = tf.cast(tf.equal(joint_hm, hmax), tf.float32)
        joint_hm = hmax * keep
        joint_hm = tf.image.resize(joint_hm, (image.shape[0], image.shape[1])) 
    
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
            joint_array = (np.ones((Config.num_joints + 1, 2)) * -1).astype(np.int) # add neck joint
            xmin, ymin, xmax, ymax = boxes[i]
            if xmax>xmin and ymax>ymin:
                crop_hm = joint_hm[:, int(ymin+0.5):int(ymax+0.5), int(xmin+0.5):int(xmax+0.5), :]
                for j in range(5, Config.num_joints): 
                    score, _, _, ys, xs = topK(crop_hm[:, :, :, j:j+1], 1)
                    if  score[0][0].numpy() > Config.joint_threshold:
                        joint_y = int((int(ymin+0.5) + ys[0][0].numpy()))
                        joint_x = int((int(xmin+0.5) + xs[0][0].numpy()))
                        joint_array[j] = joint_x, joint_y                    
                        cv2.circle(img=image, center=(joint_x, joint_y), radius=7, color=circle_color[j%2], thickness=-1)    
    
                if joint_array[5, 0] != -1 and joint_array[6, 0] != -1:
                    joint_array[17] = (joint_array[5] + joint_array[6])//2
                for joint_pair in Config.skeleton:
                    star, end = joint_pair
                    if joint_array[star, 0] == -1 or joint_array[end, 0] == -1:
                        continue
                    cv2.line(img=image, pt1=(joint_array[star]),
                                        pt2=(joint_array[end]), 
                                        color=(0, 255, 255), thickness = 3)

    return image

#%%
if __name__ == '__main__':
    
    centernet = MobileNetV2(training=True)    
    load_weights_from_epoch = Config.load_weights_from_epoch    
    centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
    # centernet.load_weights(filepath=Config.save_model_dir + "saved_model")    
    
    #%% test for video        
    from core.centernet import CenterNet, PostProcessing
    from data.dataloader import DataLoader

    videl_path = '/home/thomas_yang/Downloads/2021-09-03-kaohsiung-5g-base-vlc-record'
    video_1 = 'vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-.mp4'
    video_2 = 'vlc-record-2021-09-03-13h13m49s-rtsp___10.10.0.38_18554_fhd-.mp4' 
    video_3 = 'vlc-record-2021-09-03-13h17m52s-rtsp___10.10.0.37_28554_fhd-.mp4'
    video_4 = 'vlc-record-2021-09-03-13h23m50s-rtsp___10.10.0.25_18554_fhd-.mp4'
    
    videl_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    
    cap = cv2.VideoCapture(os.path.join(videl_path, video_3))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    half_point = length//3*2
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    save_path = '/home/thomas_yang/coco-annotator/datasets/viveland_datasets_autolabel__/'
    video_num = 'video_' + str(7) + '_'  # same as video_1 number
    total_data_list = []
    counter = 0
    
    cap_w = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('auto_label_.avi', fourcc, 20.0, (960,  540))    
    
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if counter%1 == 0:
            # print(counter//4, '/', length//4)
            save_img_name = save_path + video_num + '%08d' % (counter//4) + '.jpg'
            print(save_img_name)
            
            image_array = np.copy(frame)
            image_key = np.copy(image_array)
            image = frame[..., ::-1].astype(np.float32) / 255.
            image = cv2.resize(image, Config.get_image_size())
            image = tf.expand_dims(input=image, axis=0)        
        
            outputs = centernet(image, training=False)
            post_process = PostProcessing()
            boxes, scores, classes, bboxes_joint, scores_joint, clses_joint = post_process.testing_procedure(outputs, [image_array.shape[0], image_array.shape[1]])
            print(scores, classes)
            
            joint_hm = outputs[:,:,:, 5:5 + Config.num_joints]
            
            image_with_boxes = draw_boxes_on_image(np.copy(image_array), boxes.astype(np.int), scores, classes)
            qq = cv2.resize(image_with_boxes, (960, 540))
            
            txt_line = [save_img_name, str(image_array.shape[0]), str(image_array.shape[1])]
            keypoint_line = []
            crop_concat = np.ones((input_size, 1, 3)).astype(np.uint8)
            for box_num, box in enumerate(list(boxes)):
                for j in range(4):
                    txt_line.append(str((box + 0.5).astype(np.int)[j]))
                xmin = ((box + 0.5).astype(np.int)[0])
                ymin = ((box + 0.5).astype(np.int)[1])
                xmax = ((box + 0.5).astype(np.int)[2])
                ymax = ((box + 0.5).astype(np.int)[3])
                txt_line.append('0') # class 0 is person
                
                crop = np.copy(image_array[ymin:ymax, xmin:xmax, :])
                # cv2.imshow("crop result %d" %box_num, crop)  
                # cv2.waitKey(1)
                
                # Resize and pad the image to keep the aspect ratio and fit the expected size.
                input_image = tf.expand_dims(crop[...,::-1], axis=0)
                input_image = tf.image.resize(input_image, (input_size, input_size))
                
                # Run model inference.
                keypoint_with_scores = movenet(input_image)
                keypoint_with_scores = np.squeeze(keypoint_with_scores)
                circle_color = [(0, 0, 255), (0, 255, 0)] 
                joint_array = (np.ones((Config.num_joints + 1, 2)) * -1).astype(np.int) # add neck joint
                w = 24
                for k in range(keypoint_with_scores.shape[0]):
                    crop_joint_y = int(keypoint_with_scores[k, 0] * crop.shape[0] + ymin)
                    crop_joint_x = int(keypoint_with_scores[k, 1] * crop.shape[1] + xmin)
                    confidence = keypoint_with_scores[k, 2]
                    if confidence > 0.11:
                        cxmin = int(max(crop_joint_x-w, 0))
                        cymin = int(max(crop_joint_y-w, 0))
                        cxmax = int(min(crop_joint_x+w, 1920))
                        cymax = int(min(crop_joint_y+w, 1080))                        
                        joint_array[k] = crop_joint_x, crop_joint_y
                        keypoint_line.append(str(cxmin))
                        keypoint_line.append(str(cymin))
                        keypoint_line.append(str(cxmax))
                        keypoint_line.append(str(cymax))
                        keypoint_line.append(str(k+1))
                        if k > 5:
                            cv2.circle(img=image_with_boxes, center=(crop_joint_x, crop_joint_y), radius=5, color=circle_color[k%2], thickness=-1)
    
                if joint_array[5, 0] != -1 and joint_array[6, 0] != -1:
                    joint_array[17] = (joint_array[5] + joint_array[6])//2
                for joint_pair in Config.skeleton:
                    star, end = joint_pair
                    if joint_array[star, 0] == -1 or joint_array[end, 0] == -1 or star < 5 or end < 5:
                        continue
                    cv2.line(img=image_with_boxes, pt1=(joint_array[star]),\
                                        pt2=(joint_array[end]),\
                                        color=(0, 255, 255), thickness = 3)
            
                crop_concat = cv2.hconcat((crop_concat, cv2.resize(crop, (input_size, input_size))))

            total_data_list.append(' '.join(txt_line + keypoint_line))
                        
            crop_concat = cv2.resize(crop_concat, (crop_concat.shape[1]//2, 128))
            image_with_boxes = cv2.resize(image_with_boxes, (960, 540))           
            cv2.rectangle(crop_concat, (0 ,0), (crop_concat.shape[1], crop_concat.shape[0]), (255, 255, 255), 2)
            image_with_boxes[:128, :crop_concat.shape[1]] = crop_concat
            cv2.imshow("detect result", image_with_boxes)
            out.write(image_with_boxes)
            
            cv2.imwrite(save_img_name, image_array)
            if cv2.waitKey(1) == ord('q'):
                break
        
        counter += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open('./data_vive_land_autolabel__.txt', 'w') as fw:     
        for line in total_data_list:
            fw.write(line + '\n')  
    # total_data_list

#%%
# tmp = []
# cc = 0
# for i in range(1, 8):
#     with open('./data_vive_land_autolabel_%d.txt' %i, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             line_split_num = len(line.split(' '))
#             if line_split_num == 3:
#                 cc+=1
#                 continue
#                 print('empty', cc)
#             tmp.append(line)
            
# with open('./data_vive_land_autolabel.txt', 'w') as fw:     
#     for line in tmp:
#         fw.write(line + '\n')  




