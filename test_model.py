import tensorflow as tf
# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import os
import numpy as np
import time

from configuration import Config
from core.centernet import CenterNet, PostProcessing
from data.dataloader import DataLoader
from core.models.mobilenet import MobileNetV2
from core.centernet import Decoder
from utils.drawBboxJointLocation import draw_joint_blend_image
from utils.drawBboxJointLocation import draw_boxes_joint_on_image
from utils.drawBboxJointLocation import draw_boxes_joint_with_location_modify_on_image
from utils.drawBboxJointLocation import draw_id_on_image
from utils.drawBboxJointLocation import draw_boxes_joint_with_location_modify_on_image_speedup


def test_single_picture(picture_dir, model):
    image_array = cv2.imread(picture_dir)
    image = DataLoader.image_preprocess(is_training=False, image_dir=picture_dir)
    image = tf.expand_dims(input=image, axis=0)

    outputs = model(image, training=False)
    post_process = PostProcessing()
    
    boxes, scores, classes = post_process.testing_procedure(outputs, [image_array.shape[0], image_array.shape[1]])
    print(scores, classes)
    
    image_with_boxes = draw_boxes_joint_on_image(image_array, boxes.astype(np.int), scores, classes)
    
    return image_with_boxes

#%%
if __name__ == '__main__':
    
    centernet = MobileNetV2()    
    load_weights_from_epoch = Config.load_weights_from_epoch    
    centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
    
    # centernet = tf.keras.models.load_model(Config.save_model_dir+"epoch-{}.h5".format(load_weights_from_epoch))
    # centernet = tf.keras.models.load_model('./tmp/export.h5')
    
    #%% test for video        
    # from core.centernet import CenterNet, PostProcessing
    # from data.dataloader import DataLoader
    from configuration import Config
    # from deep_sort import DeepSort
    # deepsort = DeepSort()

    video_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MPII'
    video_1 = 'mpii_512x512.avi'

    video_path = '/home/thomas_yang/Downloads/2021-09-03-kaohsiung-5g-base-vlc-record'
    video_1 = 'vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-.mp4'
    video_2 = 'vlc-record-2021-09-03-13h13m49s-rtsp___10.10.0.38_18554_fhd-.mp4' 
    video_3 = 'vlc-record-2021-09-03-13h17m52s-rtsp___10.10.0.37_28554_fhd-.mp4'
    video_4 = 'vlc-record-2021-09-03-13h23m50s-rtsp___10.10.0.25_18554_fhd-.mp4'
    
    # video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    # video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    # video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    
    cap = cv2.VideoCapture(os.path.join(video_path, video_2))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = length//3*1
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 20.0, (960,  540))
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
                
        image_array0 = np.copy(frame)
        # image_array1 = np.copy(frame)
        
        image = frame[..., ::-1].astype(np.float32) / 255.
        image = cv2.resize(image, Config.get_image_size())
        image = tf.expand_dims(input=image, axis=0)        

        step_start_time = time.time()
        outputs = centernet(image, training=False)
        step_end_time = time.time()
        print("invoke time_cost: {:.3f}s".format(step_end_time - step_start_time))  
    
        post_process = PostProcessing()
        boxes, scores, classes, bboxes_joint, scores_joint, clses_joint = post_process.testing_procedure(outputs, [frame.shape[0], frame.shape[1]])
        print(scores, classes)
        
        boxes_tlwh = np.copy(boxes)
        boxes_tlwh[:,2:] -= boxes_tlwh[:,:2]        
        
        joint_hm = outputs[:,:,:, 5:5 + Config.num_joints]
        joint_loc = outputs[..., 5 + Config.num_joints:]
        
        image_with_boxes_joint_location_m = draw_boxes_joint_with_location_modify_on_image(image_array0, boxes.astype(np.int), scores, classes, joint_hm, joint_loc)        
        # image_with_boxes_joint_location_m = draw_boxes_joint_with_location_modify_on_image_speedup(image_array1, boxes.astype(np.int), scores, classes, joint_hm, joint_loc)
        # outputs_deepsort = deepsort.update(boxes_tlwh.tolist(), scores.tolist(), frame[..., ::-1].astype(np.float32) / 255.)
        # image_with_deep_sort = draw_id_on_image(np.copy(image_with_boxes_joint_location_m), outputs_deepsort)
        # image_with_deep_sort = cv2.resize(image_with_deep_sort, (960, 540))
        image_with_boxes_joint_location_m = cv2.resize(image_with_boxes_joint_location_m, (960, 540))

        # image_array2 = np.copy(frame)
        # image_array3 = np.copy(frame)
        # image_with_boxes_joint_location = draw_boxes_joint_on_image(image_array2, boxes.astype(np.int), scores, classes, joint_hm, joint_loc)
        # image_with_boxes = draw_joint_blend_image(image_array3, boxes.astype(np.int), scores, classes, joint_hm, True, joint_loc)                
        # image_with_boxes_joint_location = cv2.resize(image_with_boxes_joint_location, (960, 540))
        # image_with_boxes_joint_blend = cv2.resize(image_with_boxes, (960, 540))        
        # image_combine_1 = cv2.vconcat([image_with_boxes_joint_location, image_with_boxes_joint_location_m])
        # image_combine_2 = cv2.vconcat([image_with_boxes_joint_blend, image_with_deep_sort])
        # image_combine   = cv2.hconcat([image_combine_1, image_combine_2])

        # out.write(image_combine)
        # cv2.imshow("detect result", image_combine)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        
        out.write(image_with_boxes_joint_location_m)
        cv2.imshow("detect result", image_with_boxes_joint_location_m)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    tf.keras.models.save_model(centernet, filepath=Config.save_model_dir + "saved_model", include_optimizer=False, save_format="tf")

    #%% test for image
    # image_folder = '/home/thomas_yang/Downloads/vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-/vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-'
    # image_folder = './data/datasets/VOCdevkit/VOC2012/JPEGImages/'
    # image_folder = '/home/thomas_yang/ML/CenterNet_TensorFlow2/test_pictures/test_joint'
    # image_list = os.listdir(image_folder)

    # for image_dir in image_list:
    #     image_with_boxes = test_single_picture(image_dir, centernet)
    #     cv2.imshow("detect result", image_with_boxes)
    #     if cv2.waitKey(0) == ord('q'):
    #         break



