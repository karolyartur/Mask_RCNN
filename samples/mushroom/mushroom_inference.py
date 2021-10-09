from mrcnn import model as modellib, utils
from mushroom import MushroomConfig

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath("../../")

class InferenceConfig(MushroomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

MODEL_DIR = os.path.join(ROOT_DIR, 'logs', 'mushroom_model')

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR,'mushroom20211005T1305', 'mask_rcnn_mushroom_0001.h5'), by_name=True)

print('Model restored!')


dataset_path = os.path.join(ROOT_DIR,'datasets','mushroom')

img = np.array(Image.open(dataset_path + '/00000.png').getdata())
img = np.reshape(img,(1080,1920,3))

fig, ax = plt.subplots()
r = model.detect([img], verbose=1)[0]
print(r['class_ids'])
for index in range(5):
    y1, x1, y2, x2 = r['rois'][index]
    print(r['rois'][index])
    print()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, edgecolor='red', facecolor='none')
    ax.imshow(img)
    plt.axis('off')
    ax.add_patch(rect)
    plt.imshow(r['masks'][:,:,index], alpha=0.5)
    plt.axis('off')
plt.show()