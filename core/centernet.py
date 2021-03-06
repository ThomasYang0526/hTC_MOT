import tensorflow as tf
import numpy as np

from configuration import Config
from core.models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
# from core.models.mobilenet import mobilenetv2_224
from core.models.dla import dla_34, dla_60, dla_102, dla_169
from core.models.efficientdet import d0, d1, d2, d3, d4, d5, d6, d7
from data.dataloader_mot import GT
from core.loss import CombinedLoss, RegL1Loss

backbone_zoo = {"resnet_18": resnet_18(),
                "resnet_34": resnet_34(),
                "resnet_50": resnet_50(),
                "resnet_101": resnet_101(),
                "resnet_152": resnet_152(),
                "dla_34": dla_34(),
                "dla_60": dla_60(),
                "dla_102": dla_102(),
                "dla_169": dla_169(),
                "D0": d0(), "D1": d1(), "D2": d2(), "D3": d3(), "D4": d4(), "D5": d5(), "D6": d6(), "D7": d7()}
                


class CenterNet(tf.keras.Model):
    def __init__(self):
        super(CenterNet, self).__init__()
        self.backbone = backbone_zoo[Config.backbone_name]

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        x = tf.concat(values=x, axis=-1)
        return x


class PostProcessing:
    @staticmethod
    def training_procedure(batch_labels, pred):
        gt = GT(batch_labels)
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask = gt.get_gt_values()
        loss_object = CombinedLoss()
        total_loss, heatmap_loss, wh_loss, off_loss, tid_loss = loss_object(y_pred=pred, 
                                                                            heatmap_true=gt_heatmap, 
                                                                            reg_true=gt_reg, 
                                                                            wh_true=gt_wh, 
                                                                            reg_mask=gt_reg_mask, 
                                                                            indices=gt_indices,
                                                                            tid_true = gt_tid,
                                                                            tid_mask = gt_tid_mask)
        return total_loss, heatmap_loss, wh_loss, off_loss, tid_loss, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask

    @staticmethod
    def testing_procedure(pred, original_image_size):
        decoder = Decoder(original_image_size)
        detections = decoder(pred)
        bboxes = detections[0][:, 0:4]
        scores = detections[0][:, 4]
        clses = detections[0][:, 5]
        tids = detections[0][:, 6]
        embeds = detections[0][:, 6:6+512]
        return bboxes, scores, clses, tids, embeds


class Decoder:
    def __init__(self, original_image_size):
        self.K = Config.max_boxes_per_image
        self.original_image_size = np.array(original_image_size, dtype=np.float32)
        self.input_image_size = np.array(Config.get_image_size(), dtype=np.float32)
        self.downsampling_ratio = Config.downsampling_ratio
        self.score_threshold = Config.score_threshold

    def __call__(self, pred, *args, **kwargs):
        heatmap, reg, wh, embed = tf.split(value=pred[0], num_or_size_splits=[Config.num_classes, 2, 2, 512], axis=-1)
        batch_size = heatmap.shape[0]
        heatmap = Decoder.__nms(heatmap)        
        scores, inds, clses, ys, xs = Decoder.__topK(scores=heatmap, K=self.K)
        if reg is not None:
            reg = RegL1Loss.gather_feat(feat=reg, idx=inds)
            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + reg[:, :, 0:1]
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + reg[:, :, 1:2]
        else:
            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + 0.5
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + 0.5

        wh = RegL1Loss.gather_feat(feat=wh, idx=inds)
        clses = tf.cast(tf.reshape(clses, (batch_size, self.K, 1)), dtype=tf.float32)
        scores = tf.reshape(scores, (batch_size, self.K, 1))
        bboxes = tf.concat(values=[xs - wh[..., 0:1] / 2,
                                   ys - wh[..., 1:2] / 2,
                                   xs + wh[..., 0:1] / 2,
                                   ys + wh[..., 1:2] / 2], axis=2)
                     
        tid = RegL1Loss.gather_feat(feat=pred[1], idx=inds)
        tid = tf.math.argmax(tid, axis = 2)
        tid = tf.expand_dims(tid, axis=-1)
        
        embed = RegL1Loss.gather_feat(feat=embed, idx=inds)        
        
        detections = tf.concat(values=[bboxes, scores, clses, tf.cast(tid, tf.float32), embed], axis=2)    
        return [self.__map_to_original(detections)]


    def __map_to_original(self, detections):
        bboxes, scores, clses, tid, embed = tf.split(value=detections, num_or_size_splits=[4, 1, 1, 1, 512], axis=2)
        bboxes, scores, clses, tid, embed = bboxes.numpy()[0], scores.numpy()[0], clses.numpy()[0], tid.numpy()[0], embed.numpy()[0]
        resize_ratio = self.original_image_size / self.input_image_size
        bboxes[:, 0::2] = bboxes[:, 0::2] * self.downsampling_ratio * resize_ratio[1]
        bboxes[:, 1::2] = bboxes[:, 1::2] * self.downsampling_ratio * resize_ratio[0]
        bboxes[:, 0::2] = np.clip(a=bboxes[:, 0::2], a_min=0, a_max=self.original_image_size[1])
        bboxes[:, 1::2] = np.clip(a=bboxes[:, 1::2], a_min=0, a_max=self.original_image_size[0])
        score_mask = scores >= self.score_threshold        
        bboxes, scores, clses, tid, embed = [Decoder.__numpy_mask(bboxes, np.tile(score_mask, (1, 4))), 
                                             Decoder.__numpy_mask(scores, score_mask), 
                                             Decoder.__numpy_mask(clses, score_mask), 
                                             Decoder.__numpy_mask(tid, score_mask),
                                             Decoder.__numpy_mask(embed, np.tile(score_mask, (1, 512))),]
        detections = np.concatenate([bboxes, scores, clses, tid, embed], axis=-1)
        return detections

    @staticmethod
    def __numpy_mask(a, mask):
        return a[mask].reshape(-1, a.shape[-1])

    @staticmethod
    def __nms(heatmap, pool_size=7):
        hmax = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(heatmap)
        keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
        return hmax * keep

    @staticmethod
    def __topK(scores, K):
        B, H, W, C = scores.shape
        scores = tf.reshape(scores, shape=(B, -1))
        topk_scores, topk_inds = tf.math.top_k(input=scores, k=K, sorted=True)
        topk_clses = topk_inds % C
        topk_xs = tf.cast(topk_inds // C % W, tf.float32)
        topk_ys = tf.cast(topk_inds // C // W, tf.float32)
        topk_inds = tf.cast(topk_ys * tf.cast(W, tf.float32) + topk_xs, tf.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
