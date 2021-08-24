import os
import sys
import numpy as np
import scipy.io
import argparse
import logging


## Parse command line
#
#  This function parses the command line arguments
def parse_args():
    '''
    Parse command line

    This function parses the command line arguments
    
    call 'python slidebot.py --help' for command line help and usage
    '''

    parser = argparse.ArgumentParser(
        description = 'Train Mask-RCNN for slidebot dataset')
    parser.add_argument(
        '-img_key', dest='img_key',
        help='Key of the image in the .mat files, default is img',
        default='img', type=str)
    parser.add_argument(
        '-mask_key', dest='mask_key',
        help='Key of the mask in the .mat files, default is masks',
        default='masks', type=str)
    parser.add_argument(
        '-model_name', dest='model_name',
        help='Name of the model (for identification) default is slidebot_model',
        default='slidebot_model', type=str)

    args = parser.parse_args()
    return args

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

DATASET_FOLDER = os.path.join(ROOT_DIR,'datasets','slidebot', '1')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import utils


## Set logging verbosity of tensorflow
#
#  This function sets the verbosity level of tensorflow
#  @param level Logging level from the 'logging' Python package
def set_tf_loglevel(level):
    '''
    Set logging verbosity of tensorflow

    This function sets the verbosity level of tensorflow

    Inputs
        level : Logging level from the 'logging' Python package (logging.INFO, logging.WARNING ...)
    '''
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


class SlideBotConfig(Config):
    """Configuration for training on the slidebot dataset.
    Derives from the base Config class and overrides values specific
    to the slidebot dataset.
    """
    # Give the configuration a recognizable name
    NAME = "slidebot"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 3  # background + box + slide

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Don't use mini masks becuse the objects are too small
    USE_MINI_MASK = True

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1

class SlideBotDataset(utils.Dataset):
    '''
    Load slidebot dataset

    The dataset consist of multiple .mat files that each store images and corresponding masks of the scene
    '''

    def load_slidebot_data(self, count, img_key='img', mask_key='masks'):
        '''
        Load slidebot data

        Load images and masks from the .mat files
        '''
        # Add classes
        self.add_class("slidebot", 1, "box")
        self.add_class("slidebot", 2, "slide")

        # Store img and mask keys:
        self.img_key = img_key
        self.mask_key = mask_key

        # Add images
        random_samples = np.random.choice(np.array(os.listdir(DATASET_FOLDER)), count)
        for sample_name in random_samples:
            sample_path = os.path.join(DATASET_FOLDER, sample_name)
            sample_id = int(sample_name.split('.')[0])
            self.add_image('slidebot', sample_id, sample_path)

    def load_image(self, image_id):
        '''
        Overwrite of the load_image function in base class

        Load image from the .mat files
        '''
        mat = scipy.io.loadmat(self.image_info[image_id]['path'])
        img = mat[self.img_key]*255
        return img.astype(np.uint8)

    def load_mask(self, image_id):
        '''
        Overwrite of the load_mask function in base class

        Load mask from the .mat files
        '''
        mat = scipy.io.loadmat(self.image_info[image_id]['path'])
        class_ids = np.ones(mat[self.mask_key].shape[2])*2
        class_ids[0] = 1
        return mat[self.mask_key].astype(np.uint8), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        '''
        Overwrite of the image_reference function in base class

        Return the image path
        '''
        return self.image_info[image_id]['path']


if __name__ == "__main__":

    args = parse_args()

    set_tf_loglevel(logging.FATAL)

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, 'logs', args.model_name)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'logs', 'mask_rcnn_coco.h5')

    config = SlideBotConfig()
    config.display()

    # Training dataset
    dataset_train = SlideBotDataset()
    dataset_train.load_slidebot_data(1, args.img_key, args.mask_key)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SlideBotDataset()
    dataset_val.load_slidebot_data(1, args.img_key, args.mask_key)
    dataset_val.prepare()


    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

    # Load COCO pre-trained weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head layers
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')