import cv2
from test_model import test_single_picture
from configuration import Config


def visualize_training_results(pictures, model, epoch):
    """
    :param pictures: List of image directories.
    :param model:
    :param epoch:
    :return:
    """
    index = 0
    for picture in pictures:
        index += 1
        result = test_single_picture(picture_dir=picture, model=model)
        cv2.imwrite(filename=Config.training_results_save_dir + "picture-{}.jpg".format(index), img=result)
        # cv2.imwrite(filename=Config.training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)


def visualize_training_results_step(pictures, model, epoch, step):
    """
    :param pictures: List of image directories.
    :param model:
    :param epoch:
    :return:
    """
    index = 0
    for picture in pictures:
        index += 1
        print(picture)
        result = test_single_picture(picture_dir=picture, model=model)
        cv2.imwrite(filename=Config.training_results_save_dir + "picture-{}.jpg".format(index), img=result)
        # cv2.imwrite(filename=Config.training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)
        # cv2.imwrite(filename=Config.training_results_save_dir + "epoch-{}-step-{}-picture-{}.jpg".format(epoch, step, index), img=result)