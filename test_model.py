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
from core.models.resnetFPN import Res50FPN
from core.centernet import Decoder
from utils.drawBboxJointLocation import draw_boxes_ReID_on_image
# from utils.drawBboxJointLocation import draw_boxes_joint_on_image
from utils.drawBboxJointLocation import draw_boxes_joint_with_location_modify_on_image
# from utils.drawBboxJointLocation import draw_id_on_image
# from utils.drawBboxJointLocation import draw_boxes_joint_with_location_modify_on_image_speedup
from scipy.optimize import linear_sum_assignment

def test_single_picture(picture_dir, model):    
    image_array = cv2.imread(picture_dir)
    image = DataLoader.image_preprocess(is_training=False, image_dir=picture_dir)
    image = tf.expand_dims(input=image, axis=0)

    outputs = model(image, training=False)
    post_process = PostProcessing()
    
    boxes, scores, classes, tids, embeds = post_process.testing_procedure(outputs, [image_array.shape[0], image_array.shape[1]])
    print(scores, classes, tids)
    
    image_with_boxes = draw_boxes_ReID_on_image(image_array, boxes.astype(np.int), scores, classes, tids)
    
    return image_with_boxes

#%%
if __name__ == '__main__':
    
    centernet = Res50FPN()    
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

    # video_path = '/home/thomas_yang/Downloads/2021-09-03-kaohsiung-5g-base-vlc-record'
    # video_1 = 'vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-.mp4'
    # video_2 = 'vlc-record-2021-09-03-13h13m49s-rtsp___10.10.0.38_18554_fhd-.mp4' 
    # video_3 = 'vlc-record-2021-09-03-13h17m52s-rtsp___10.10.0.37_28554_fhd-.mp4'
    # video_4 = 'vlc-record-2021-09-03-13h23m50s-rtsp___10.10.0.25_18554_fhd-.mp4'
    
    video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    
    cap = cv2.VideoCapture(os.path.join(video_path, video_2))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = length//4.5*1
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 20.0, (960,  540))
    
    embed_0 = []
    embed_1 = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        count += 1        
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
        boxes, scores, classes, tids, embeds = post_process.testing_procedure(outputs, [frame.shape[0], frame.shape[1]])        
        
        tmp = []
        cost = np.zeros((2, 2))
        for embed_idx, embed in enumerate(embeds):
            embed = np.expand_dims(embed, 1).transpose()
            embed = cv2.resize(embed, (256, 4), interpolation = cv2.INTER_NEAREST)
            
            if embed_0 == []:
                embed_0 = embed
                continue
            if embed_0.shape[0]!=1 and embed_1 == []:
                embed_1 = embed
                continue
            
            diff0 = np.sum((embed - embed_0[-1])**2)
            diff1 = np.sum((embed - embed_1[-1])**2)
            
            cost[0, embed_idx] = diff0
            cost[1, embed_idx] = diff1
            
            tmp.append(embed)

        if count == 1:
            continue
        
        row_ind, col_ind = linear_sum_assignment(cost)            
        embed_0 = cv2.vconcat([embed_0, tmp[row_ind[0]]])
        embed_1 = cv2.vconcat([embed_1, tmp[row_ind[1]]])
        
        cv2.imshow("embed_idx %d" %0, embed_0)
        cv2.imshow("embed_idx %d" %1, embed_1)
        
        print(scores, classes)
        
        image_with_boxes_joint_location_m = draw_boxes_ReID_on_image(image_array0, boxes.astype(np.int), scores, classes, tids)
        image_with_boxes_joint_location_m = cv2.resize(image_with_boxes_joint_location_m, (960, 540))
        
        out.write(image_with_boxes_joint_location_m)
        cv2.imshow("detect result", image_with_boxes_joint_location_m)
        if cv2.waitKey(0) == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # tf.keras.models.save_model(centernet, filepath=Config.save_model_dir + "saved_model", include_optimizer=False, save_format="tf")

    #%% test for image
    # image_folder = '/home/thomas_yang/ML/hTC_MOT/test_pictures/test_sample_1/'
    # image_list = [image_folder+i for i in os.listdir(image_folder)]
    # image_list.sort()
                      
    # for image_dir in image_list:             
    #     image_with_boxes = test_single_picture(image_dir, centernet)
    #     # image_with_boxes = cv2.resize(image_with_boxes, (960, 540))
    #     cv2.imshow("detect result", image_with_boxes)
    #     if cv2.waitKey(0) == ord('q'):                
    #         break
    # cv2.destroyAllWindows()



