from mrcnn import model as modellib, utils
from slidebot import SlideBotConfig

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath("../../")

class InferenceConfig(SlideBotConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

MODEL_DIR = os.path.join(ROOT_DIR, 'logs', 'slidebot_model')

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR,'slidebot20210828T1001', 'mask_rcnn_slidebot_0010.h5'), by_name=True)

print('Model restored!')


dataset_path = os.path.join(ROOT_DIR,'datasets','slidebot','1')
for matfile in os.listdir(dataset_path):
    if matfile == '0010.mat':
        fig, ax = plt.subplots()
        filepath = os.path.join(dataset_path, matfile)

        mat = scipy.io.loadmat(filepath)
        img = mat['img']
        img = img.astype(np.uint8)
        r = model.detect([img], verbose=1)[0]

        # box_index = list(np.argwhere(r['class_ids']==1))

        counter = 0
        for index in r['class_ids']:

            # if box_index:
                # box_id = box_index[0][0]

                # y1, x1, y2, x2 = r['rois'][box_id]
            y1, x1, y2, x2 = r['rois'][index]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, edgecolor='red', facecolor='none')

            ax.imshow(img)
            plt.axis('off')
            plt.savefig(matfile.split('.')[0] + '.png')

            ax.add_patch(rect)

            print(r['class_ids'])

            plt.imshow(r['masks'][:,:,index], alpha=0.5)
            plt.axis('off')
            plt.savefig(matfile.split('.')[0] + str(counter) + '_overlay.png')
            counter += 1