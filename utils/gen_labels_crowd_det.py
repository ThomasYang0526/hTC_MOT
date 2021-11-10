import os.path as osp
import os
import cv2
import json
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_crowd(data_root, label_root, ann_root):
    mkdirs(label_root)
    anns_data = load_func(ann_root)

    tid_curr = 0
    total_info = []
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        
        info = [img_path, str(img_height), str(img_width)]        
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            xmin, ymin, w, h = anns[i]['fbox']
            xmax = xmin + w / 2
            ymax = ymin + h / 2
            # label_fpath = img_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            # label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
            
            info.append(str(int(xmin)))
            info.append(str(int(ymin)))
            info.append(str(int(xmax)))
            info.append(str(int(ymax)))
            info.append(str(int(0)))
        info_str = ' '.join(info)
        total_info.append(info_str)
        print(info_str) 
        
    label_fpath = '/home/thomas_yang/ML/hTG_ObjectDetection/txt_file/pose_detection/crowdhuman.txt'
    with open(label_fpath, 'a') as f:
        for label_str in total_info:
            f.write(label_str)
    tid_curr += 1


if __name__ == '__main__':
    data_val = '/home/thomas_yang/ML/datasets/crowdhuman/images/val'
    label_val = '/home/thomas_yang/ML/datasets/crowdhuman/labels_with_ids/val'
    ann_val = '/home/thomas_yang/ML/datasets/crowdhuman/annotation_val.odgt'
    fname_val = '/home/thomas_yang/ML/hTG_ObjectDetection/txt_file/pose_detection/crowdhuman_val.txt'
    data_train = '/home/thomas_yang/ML/datasets/crowdhuman/images/train'
    label_train = '/home/thomas_yang/ML/datasets/crowdhuman/labels_with_ids/train'
    ann_train = '/home/thomas_yang/ML/datasets/crowdhuman/annotation_train.odgt'
    fname_train = '/home/thomas_yang/ML/hTG_ObjectDetection/txt_file/pose_detection/crowdhuman_train.txt'
    # gen_labels_crowd(data_train, label_train, ann_train)
    # gen_labels_crowd(data_val, label_val, ann_val)

    # data_root, label_root, ann_root, fname_root = data_train, label_train, ann_train, fname_train
    data_root, label_root, ann_root, fname_root = data_val, label_val, ann_val, fname_val

    mkdirs(label_root)
    anns_data = load_func(ann_root)

    total_info = []
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        
        info = [img_path, str(img_height), str(img_width)]        
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            xmin, ymin, w, h = anns[i]['fbox']
            xmax = xmin + w
            ymax = ymin + h

            info.append(str(int(xmin)))
            info.append(str(int(ymin)))
            info.append(str(int(xmax)))
            info.append(str(int(ymax)))
            info.append(str(int(0)))
        info_str = ' '.join(info)
        info_str += '\n'
        total_info.append(info_str)
        # print(info_str) 
        
    label_fpath = fname_root
    with open(label_fpath, 'w') as f:
        for label_str in total_info:
            f.write(label_str)

