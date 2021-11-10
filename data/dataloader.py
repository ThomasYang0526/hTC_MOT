import tensorflow as tf
import numpy as np

from configuration import Config
from utils.gaussian import gaussian_radius, draw_umich_gaussian

from PIL import Image
import cv2

# import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import sys
import time
import pickle

class DetectionDataset:
    def __init__(self):
        self.txt_file = Config.txt_file_dir
        self.val_txt_file = Config.val_txt_file_dir
        self.batch_size = Config.batch_size
        self.val_batch_size = Config.batch_size

    @staticmethod
    def __get_length_of_dataset(dataset):
        length = 0
        for _ in dataset:
            length += 1
        return length

    def generate_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        dataset = dataset.shuffle(length_of_dataset) # shuffle
        print('Total data number: %d images' %length_of_dataset)
        train_dataset = dataset.batch(batch_size=self.batch_size)
        return train_dataset, length_of_dataset

    def generate_val_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.val_txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        val_dataset = dataset.batch(batch_size=self.val_batch_size)
        return val_dataset, length_of_dataset

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class DataLoader:

    input_image_height = Config.get_image_size()[0]
    input_image_width = Config.get_image_size()[1]
    input_image_channels = Config.image_channels

    def __init__(self):
        self.max_boxes_per_image = Config.max_boxes_per_image
        self.seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),
            # iaa.GaussianBlur(sigma=(0, 1)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255)),
            iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)}, scale=(0.6, 1.2)),
            iaa.Affine(rotate=(-10, 10)),
            iaa.AddToHueAndSaturation((-10, 10), per_channel=True),
            iaa.GammaContrast((0.75, 1.25)),
            # iaa.MultiplyHueAndSaturation(),
            iaa.MotionBlur(k=3),
            # iaa.Crop(percent=(0, 0.1)),
        ], random_order=True)  
        
        self.flip = iaa.Sequential([iaa.Fliplr(1.0)])


    def read_batch_data(self, batch_data, augment):
        batch_size = batch_data.shape[0]
        image_file_list = []
        boxes_list = []
        
        for n in range(batch_size):
            image_file, boxes = self.__get_image_information(single_line=batch_data[n])
            image_file_list.append(image_file)
            boxes_list.append(boxes)        
        boxes = np.stack(boxes_list, axis=0)           
        boxes = np.stack(boxes_list, axis=0)
        
        image_tensor_list = []
        for idx, image in enumerate(image_file_list):
            if augment == False:
                image_tensor = self.image_preprocess(is_training=True, image_dir=image)
            else:
                image_tensor, boxes_ = self.image_preprocess_augmentation(is_training=True, image_dir=image, boxes=boxes[idx])
                boxes[idx] = boxes_
            image_tensor_list.append(image_tensor)        
        images = tf.stack(values=image_tensor_list, axis=0)        
        return images, boxes

    def __get_image_information(self, single_line):
        """
        :param single_line: tensor
        :return:
        image_file: string, image file dir
        boxes_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
        """
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        line_list = line_string.strip().split(" ")
        image_file, image_height, image_width = line_list[:3]
        image_height, image_width = int(float(image_height)), int(float(image_width))
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * 5]))
                ymin = int(float(line_list[3 + index * 5 + 1]))
                xmax = int(float(line_list[3 + index * 5 + 2]))
                ymax = int(float(line_list[3 + index * 5 + 3]))
                class_id = int(line_list[3 + index * 5 + 4])
                xmin, ymin, xmax, ymax = DataLoader.box_preprocess(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        boxes_array = np.array(boxes, dtype=np.float32)
        # step_end_time = time.time()
        # print('----get_image_information', step_end_time - step_start_time)          
        return image_file, boxes_array           

    @classmethod
    def box_preprocess(cls, h, w, xmin, ymin, xmax, ymax):
        resize_ratio = [DataLoader.input_image_height / h, DataLoader.input_image_width / w]
        xmin = int(resize_ratio[1] * xmin)
        xmax = int(resize_ratio[1] * xmax)
        ymin = int(resize_ratio[0] * ymin)
        ymax = int(resize_ratio[0] * ymax)
        return xmin, ymin, xmax, ymax

    @classmethod
    def image_preprocess(cls, is_training, image_dir):
        image_raw = tf.io.read_file(filename=image_dir)
        decoded_image = tf.io.decode_image(contents=image_raw, channels=DataLoader.input_image_channels, dtype=tf.dtypes.float32)
        decoded_image = tf.image.resize(images=decoded_image, size=(DataLoader.input_image_height, DataLoader.input_image_width))
        return decoded_image
    
    # @classmethod
    def image_preprocess_augmentation(self, is_training, image_dir, boxes, jitter=.3, hue=.1, sat=1.5, val=1.5):
        
        image = Image.open(image_dir)
        iw, ih = image.size
        h, w = Config.get_image_size()
        
        decoded_image = np.array(image.resize((w, h), Image.BILINEAR))
        if len(decoded_image.shape) == 2:
            decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_GRAY2RGB)
        else:
            if decoded_image.shape[2] != 3:
                decoded_image = decoded_image[:, :, 0:3]

#%%        
        # boxes = labels[4]
        # decoded_image = (images[4].numpy()*255).astype(np.uint8)
        boxes_au = []
        for box_idx in range(boxes.shape[0]):
            if boxes[box_idx,4] == -1:
                break
            boxes_au.append(BoundingBox(x1=boxes[box_idx, 0],y1=boxes[box_idx,1],x2=boxes[box_idx,2],y2=boxes[box_idx,3], label=boxes[box_idx,4]))
                
        bbs = BoundingBoxesOnImage(boxes_au, shape=Config.get_image_size())
        
        seq = iaa.Sometimes(0.7, self.seq)
        image_aug, bbs_aug = seq(image=decoded_image, bounding_boxes=bbs)
        
        flip_flag = False
        if np.random.rand() > 0.5:
            image_aug, bbs_aug = self.flip(image=image_aug, bounding_boxes=bbs_aug)
            flip_flag = True
            
        image_aug = np.array(image_aug).astype(np.float32)/255.
        
        bbs_aug = bbs_aug.to_xyxy_array()
        # bbs_aug = bbs_aug.clip_out_of_image()
        # ia.imshow(bbs_aug.draw_on_image(image_aug))
        
        box_data = np.copy(boxes)
        box_data[:bbs_aug.shape[0],:4] = bbs_aug
        for box_idx in range(box_data.shape[0]):
            if box_data[box_idx, 0] < 0: box_data[box_idx, 0] = 0
            if box_data[box_idx, 1] < 0: box_data[box_idx, 1] = 0
            if box_data[box_idx, 2] >= DataLoader.input_image_width: box_data[box_idx, 2] = DataLoader.input_image_width
            if box_data[box_idx, 3] >= DataLoader.input_image_height: box_data[box_idx, 3] = DataLoader.input_image_height
        
        if flip_flag:
            for box_idx in range(box_data.shape[0]):
                if box_data[box_idx, -1] > 0 and box_data[box_idx, -1] %2 == 0:
                    box_data[box_idx, -1] += 1
                elif box_data[box_idx, -1] > 0 and box_data[box_idx, -1] %2 == 1:
                    box_data[box_idx, -1] -= 1
                    
#%%     
        return image_aug, box_data

class GT:
    def __init__(self, batch_labels):
        self.downsampling_ratio = Config.downsampling_ratio # efficientdet: 8, others: 4
        self.features_shape = np.array(Config.get_image_size(), dtype=np.int32) // self.downsampling_ratio # efficientnet: 64*64
        self.batch_labels = batch_labels
        self.batch_size = batch_labels.shape[0]

    def get_gt_values(self):
        gt_heatmap = np.zeros(shape=(self.batch_size, self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        gt_reg = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_wh = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_reg_mask = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        gt_indices  = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
             
        for i, label in enumerate(self.batch_labels):
            label = label[label[:, 4] != -1]
            
            # object class part
            label_class = label[label[:, 4] == 0]
            hm, reg, wh, reg_mask, ind = self.__decode_label(label_class)
            gt_heatmap[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind
                        
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def __decode_label(self, label):
        hm = np.zeros(shape=(self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        reg = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)
        wh = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)
        reg_mask = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)
        ind = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)
        radius_list = []
        for j, item in enumerate(label): #原始座標
            item_down = item[:4] / self.downsampling_ratio #原始座標/縮小比例(8)
            xmin, ymin, xmax, ymax = item_down
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(self.features_shape[1], xmax)
            ymax = min(self.features_shape[0], ymax)
            
            class_id = item[4].astype(np.int32)
            # h, w = int(ymax - ymin), int(xmax - xmin)
            # radius = gaussian_radius((h, w))
            h, w = ymax - ymin, xmax - xmin
            radius = gaussian_radius((int(h), int(w)))
            radius = max(0, int(radius))   
            radius_list.append(radius)
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)
            center_point_int = center_point.astype(np.int32)
            draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            reg[j] = center_point - center_point_int
            # print(reg[j], center_point, center_point_int, xmin, ymin, xmax, ymax)
            wh[j] = 1. * w, 1. * h
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]            
        return hm, reg, wh, reg_mask, ind

if __name__ == '__main__':
    train_dataset = DetectionDataset()
    train_data, train_size = train_dataset.generate_datatset()    
    data_loader = DataLoader()
    steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)
    for step, batch_data in enumerate(train_data): 
        if step != 302:
            continue
        print(step, '------------------------', train_size)
        # load data 
        images, labels = data_loader.read_batch_data(batch_data, augment = False)        
        img_rgb = np.array((images.numpy()*255).astype(np.uint8)[0][...,::-1])
        
        gt = GT(labels)
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices = gt.get_gt_values()

        B, H, W, C = gt_heatmap.shape
        indice = gt_indices[gt_reg_mask==1]
        mask_gt_wh = gt_wh[gt_reg_mask==1]

        topk_xs = (indice % W).astype(np.int)
        topk_ys = (indice // W).astype(np.int)
        
        for i in range(topk_xs.shape[0]):            
            cv2.circle(img_rgb, (topk_xs[i]*4, topk_ys[i]*4), 4, (0, 255, 0), -1)
            for j in range(topk_xs.shape[0]):
                xmin = int(-mask_gt_wh[i,0]*4/2+topk_xs[i]*4)
                ymin = int(-mask_gt_wh[i,1]*4/2+topk_ys[i]*4)
                xmax = int(mask_gt_wh[i,0]*4/2+topk_xs[i]*4)
                ymax = int(mask_gt_wh[i,1]*4/2+topk_ys[i]*4)
                cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
        
        cv2.imshow('img', img_rgb)
        cv2.imshow('gt_heatmap', cv2.resize(gt_heatmap[0], (416, 416)))
        if cv2.waitKey(0) == ord('q'):
            break        

    cv2.destroyAllWindows()