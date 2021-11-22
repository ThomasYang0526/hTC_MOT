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
from core.models.resnetFPN import Res50FPN, Res50FPN_reID
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

    outputs = model([image, np.zeros((1, Config.max_boxes_per_image, 1))], training=False)
    post_process = PostProcessing()
    
    boxes, scores, classes, tids, embeds = post_process.testing_procedure(outputs, [image_array.shape[0], image_array.shape[1]])
    print(scores, classes, tids)
    
    image_with_boxes = draw_boxes_ReID_on_image(image_array, boxes.astype(np.int), scores, classes, tids)
    
    return image_with_boxes

#%%
if __name__ == '__main__':
    
    centernet = Res50FPN_reID()    
    load_weights_from_epoch = Config.load_weights_from_epoch    
    centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
    
    # centernet = tf.keras.models.load_model(Config.save_model_dir+"epoch-{}.h5".format(load_weights_from_epoch))
    # centernet = tf.keras.models.load_model('./tmp/export.h5')
    
    #%% test for video        
    # from core.centernet import CenterNet, PostProcessing
    # from data.dataloader import DataLoader
    from configuration import Config
    import collections
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
    
    cap = cv2.VideoCapture(os.path.join(video_path, video_5))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = length//7*1
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 20.0, (2000,  540))
    
    embed_0 = np.zeros((540,512), dtype=np.float32)
    embed_1 = np.zeros((540,512), dtype=np.float32)
    inter = (np.ones((540,16,3))*(0,128,128)).astype(np.uint8)
    count = 0          
    xy0 = np.zeros((1,2))
    xy1 = np.zeros((1,2))
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
        outputs = centernet([image, np.zeros((1, Config.max_boxes_per_image, 1))], training=False)

        step_end_time = time.time()
        print("invoke time_cost: {:.3f}s".format(step_end_time - step_start_time))  
    
        post_process = PostProcessing()
        boxes, scores, classes, tids, embeds = post_process.testing_procedure(outputs, [frame.shape[0], frame.shape[1]])  
        image_with_boxes_joint_location_m = draw_boxes_ReID_on_image(image_array0, boxes.astype(np.int), scores, classes, tids)
                
        argidx = np.argsort(boxes[:, 0])
        # print(boxes, boxes.shape)
        # boxes  = boxes[argidx]
        # boxes_ = boxes[-2:,:]
        
        # xy0 = (boxes_[0][2:] + boxes_[0][0:2])//2
        # xy1 = (boxes_[1][2:] + boxes_[1][0:2])//2

        xy0 = (boxes[argidx[-2]][2:] + boxes[argidx[-2]][0:2])//2
        xy1 = (boxes[argidx[-1]][2:] + boxes[argidx[-1]][0:2])//2
                  
        diff0 = np.copy(xy0[0])
        diff1 = np.copy(xy1[0])
        if diff0 < diff1:
            embed_0[count] = embeds[argidx[-2]].transpose()
            embed_1[count] = embeds[argidx[-1]].transpose()
            cv2.circle(img=image_array0, center=(int(xy0[0]), int(xy0[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=image_array0, center=(int(xy1[0]), int(xy1[1])), radius=5, color=(0, 255, 0), thickness=-1)
        else :
            embed_0[count] = embeds[argidx[-1]].transpose()
            embed_1[count] = embeds[argidx[-2]].transpose()
            cv2.circle(img=image_array0, center=(int(xy1[0]), int(xy1[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=image_array0, center=(int(xy0[0]), int(xy0[1])), radius=5, color=(0, 255, 0), thickness=-1)        

        
        # cv2.imshow("embed_idx %d" %0, embed_0)
        # cv2.imshow("embed_idx %d" %1, embed_1)        
        
        print(scores, classes, count, diff0, diff1)
        count += 1 
        

        image_with_boxes_joint_location_m = cv2.resize(image_with_boxes_joint_location_m, (960, 540))
        image_with_boxes_joint_location_m = cv2.hconcat([image_with_boxes_joint_location_m, 
                                                         cv2.cvtColor((embed_0/5*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                                         inter,
                                                         cv2.cvtColor((embed_1/5*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)])
        
        out.write(image_with_boxes_joint_location_m)
        cv2.imshow("detect result", image_with_boxes_joint_location_m)
        if cv2.waitKey(1) == ord('q'):
            break
        
        if count == 420:
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

#%%
    import matplotlib.pyplot as plt
    diff_00 = []
    diff_01 = []
    diff_10 = []
    diff_11 = []
    # for i in range(2, embed_0.shape[0]):
    for i in range(2, 419):
        diff_00.append(np.sqrt(np.sum((embed_0[i, :] - embed_0[i-1, :])**2))/embed_0.shape[1])
        diff_01.append(np.sqrt(np.sum((embed_0[i, :] - embed_1[i-1, :])**2))/embed_0.shape[1])
        diff_10.append(np.sqrt(np.sum((embed_1[i, :] - embed_0[i-1, :])**2))/embed_0.shape[1])
        diff_11.append(np.sqrt(np.sum((embed_1[i, :] - embed_1[i-1, :])**2))/embed_0.shape[1])
        
    diff_00 = np.array(diff_00)
    diff_01 = np.array(diff_01)
    diff_10 = np.array(diff_10)
    diff_11 = np.array(diff_11)
    
    x = np.arange(0, len(diff_00))
    plt.figure(0)
    plt.plot(x, diff_00, marker="o")
    plt.plot(x, diff_10, marker="o")
    plt.xlabel("Time Frame")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(x, diff_01, marker="o")
    plt.plot(x, diff_11, marker="o")
    plt.xlabel("Time Frame")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()



























