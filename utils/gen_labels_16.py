import os.path as osp
import os
import numpy as np
import collections 


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/home/thomas_yang/ML/datasets/MOT16/images/train'
label_root = '/home/thomas_yang/ML/datasets/MOT16/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1

total_info = collections.defaultdict(list)
# total_info = []
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    img_path = osp.join(seq_root, seq, 'img1')
    info = [img_path, str(seq_height), str(seq_width)]   

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
            
        # x += w / 2
        # y += h / 2
        # label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        img_fpath = osp.join(img_path, '{:06d}.jpg'.format(fid)) + ' ' + str(seq_height) + ' ' +  str(seq_width)
        
        # label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        # with open(label_fpath, 'a') as f:
            # f.write(label_str)
        
        xmin = x
        ymin = y
        xmax = xmin + w
        ymax = ymin + h
        
        label_str = '%s %s %s %s 0' %(str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax)))
        total_info[img_fpath].append(label_str)
        
label_fpath = '/home/thomas_yang/ML/hTG_ObjectDetection/txt_file/pose_detection/MOT_16.txt'
with open(label_fpath, 'w') as f:
    for key in total_info:
        label_str = key + ' ' + ' '.join(total_info[key]) + '\n'
        f.write(label_str)








