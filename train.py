import tensorflow as tf
# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from configuration import Config
import cv2            
import time
import numpy as np
from core.centernet import PostProcessing
from utils.visualize import visualize_training_results, visualize_training_results_step
import datetime
from data.dataloader_mot import DetectionDataset, DataLoader
from core.models.mobilenet import MobileNetV2
# from core.models.dla_34 import DLA_MODEL
from core.models.resnetFPN import Res50FPN
from utils.show_traing_val_image import show_traing_val_image

if __name__ == '__main__':

    # train/validation dataset
    train_dataset = DetectionDataset()
    train_data, train_size = train_dataset.generate_datatset()
    val_dataset = DetectionDataset()
    val_data, val_size = val_dataset.generate_val_datatset()
    
    data_loader = DataLoader()
    steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)
    
    # save logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    # model
    # centernet = MobileNetV2(training=True)
    # centernet = MobileNetV2()
    # centernet = DLA_MODEL()
    centernet = Res50FPN()
    
    # load pre-train weight
    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        # centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}.h5".format(load_weights_from_epoch), by_name=True)
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1
        
   

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                 decay_steps=Config.learning_rate_decay_step,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # metrics
    lr_metric = tf.metrics.Mean('learning_rate', dtype=tf.float32)
    total_loss_metric = tf.metrics.Mean('total_loss', dtype=tf.float32)
    heatmap_loss_metric = tf.metrics.Mean('heatmap_loss', dtype=tf.float32)
    wh_loss_metric = tf.metrics.Mean('wh_loss', dtype=tf.float32)    
    offset_loss_metric = tf.metrics.Mean('offset_loss', dtype=tf.float32)
    tid_loss_metric = tf.metrics.Mean('tid_loss', dtype=tf.float32)
    val_loss_metric = tf.metrics.Mean('val_loss', dtype=tf.float32)
    
    post_process = PostProcessing()
    
    def train_step(model, batch_images, batch_labels, current_train_step, show_data=False):
        with tf.GradientTape() as tape:
            pred = model(batch_images, True)
            total_loss, heatmap_loss, wh_loss, offset_loss, \
            tid_loss, \
            gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, \
            gt_tid, gt_tid_mask \
            = post_process.training_procedure(batch_labels=batch_labels, pred=pred)
        gradients = tape.gradient(target=total_loss, sources=model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        # update metric
        lr_metric.update_state(values=optimizer._decayed_lr(tf.float32).numpy())
        total_loss_metric.update_state(values=total_loss)
        heatmap_loss_metric.update_state(values=heatmap_loss)
        wh_loss_metric.update_state(values=wh_loss)
        offset_loss_metric.update_state(values=offset_loss)
        tid_loss_metric.update_state(values=tid_loss)
        return pred, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask

    def val_step(model, batch_images, batch_labels, current_train_step, epoch):  
        pred = model(batch_images, False)
        total_loss, heatmap_loss, wh_loss, offset_loss, \
        tid_loss, \
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, \
        gt_tid, gt_tid_mask\
        = post_process.training_procedure(batch_labels=batch_labels, pred=pred)
        
        # update metric
        val_loss_metric.update_state(values=total_loss)
        return pred, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask
    
    #%%
    for epoch in range(load_weights_from_epoch + 1, Config.epochs):
        # training part
        for step, batch_data in enumerate(train_data):
                        
            current_train_step = epoch * (train_size // Config.batch_size) + step
            
            # load data 
            step_start_time = time.time()
            images, labels = data_loader.read_batch_data(batch_data, augment= False)
            step_end_time = time.time()
            load_image_time = step_end_time - step_start_time
            # print("load image time_cost: {:.3f}s".format(load_image_time))
                     
            # train step
            step_start_time = time.time()
            pred, gt_heatmap, _, _, _, gt_indices, gt_tid, gt_tid_mask = train_step(centernet, images, labels, current_train_step, show_data=False)
            step_end_time = time.time()
            training_image_time = step_end_time - step_start_time
            # print("training time_cost: {:.3f}s".format(training_image_time))

            # validation part
            if (step+1) % Config.val_images_during_training_step_save_frequency == 0:             
                for val_batch_data in val_data:
                    val_images, val_labels = data_loader.read_batch_data(val_batch_data, augment= False)  
                    val_pred, val_gt_heatmap, _, _, _, _, val_tid, val_tid_mask = val_step(centernet, val_images, val_labels, current_train_step, epoch)
                    
                    # show gt_heatmap and pre_heatpmap 
                    show_traing_val_image(val_images, val_gt_heatmap, val_pred, training = False)
                    print('val_loss:', val_loss_metric.result().numpy())
            
                with val_summary_writer.as_default():
                    tf.summary.scalar('summary/total_loss', val_loss_metric.result(), step = current_train_step)
                val_loss_metric.reset_states()

            
            # save logs
            with train_summary_writer.as_default():
                tf.summary.scalar('summary/total_loss', total_loss_metric.result(), step = current_train_step)
                tf.summary.scalar('summary/heatmap_loss', heatmap_loss_metric.result(), step = current_train_step)
                tf.summary.scalar('summary/wh_loss', wh_loss_metric.result(), step = current_train_step)
                tf.summary.scalar('summary/offset_loss', offset_loss_metric.result(), step = current_train_step)
                tf.summary.scalar('summary/tid_loss', tid_loss_metric.result(), step = current_train_step)
                tf.summary.scalar('summary/lr', lr_metric.result(), step = current_train_step)
            
            # save test image and model in step during training
            if (step+1) % Config.test_images_during_training_step_save_frequency == 0:
                # centernet.save_weights(filepath=Config.save_model_dir+"epoch-{}".format(epoch), save_format="tf")
                # centernet.save_weights(filepath=Config.save_model_dir+"epoch-{}.h5".format(epoch), save_format="h5")
                visualize_training_results_step(pictures=Config.test_images_dir_list, model=centernet, epoch=epoch, step=1)
                        
            print("Epoch: {}/{}, step: {}/{}, time_cost: load {:.3f}s, train {:.3f}s, total {:.3f}s".format(epoch,
                                                                                                            Config.epochs,
                                                                                                            step,
                                                                                                            steps_per_epoch,
                                                                                                            load_image_time,
                                                                                                            training_image_time,
                                                                                                            load_image_time + training_image_time))
            

            print("Total loss: %s, heatmap: %s, wh: %s, offset: %s, tid: %s" %(str(np.round(total_loss_metric.result()  ,   3)),
                                                                               str(np.round(heatmap_loss_metric.result(),   3)),
                                                                               str(np.round(wh_loss_metric.result()     ,   3)),
                                                                               str(np.round(offset_loss_metric.result() ,   3)),
                                                                               str(np.round(tid_loss_metric.result() ,      3))))

            lr_metric.reset_states()
            total_loss_metric.reset_states()
            heatmap_loss_metric.reset_states()
            wh_loss_metric.reset_states()
            offset_loss_metric.reset_states()
            tid_loss_metric.reset_states()
            
            # show gt_heatmap and pre_heatpmap
            show_traing_val_image(images, gt_heatmap, pred, training = True)

        # save weight part
        if epoch % Config.save_frequency == 0:
            centernet.save_weights(filepath=Config.save_model_dir+"epoch-{}".format(epoch), save_format="tf")
            centernet.save_weights(filepath=Config.save_model_dir+"epoch-{}.h5".format(epoch), save_format="h5")

        # save test image part
        if Config.test_images_during_training:
            visualize_training_results(pictures=Config.test_images_dir_list, model=centernet, epoch=epoch)    

    centernet.save_weights(filepath=Config.save_model_dir + "saved_model", save_format="tf")
    # tf.keras.models.save_model(centernet, filepath=Config.save_model_dir + "saved_model", include_optimizer=False, save_format="tf")
