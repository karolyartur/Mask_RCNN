import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mat = scipy.io.loadmat('../../datasets/slidebot/1/0001.mat')
ys, xs = np.where(mat['masks'][:,:,0]==1)
print(xs.min(),xs.max(),ys.min(),ys.max())

fig, ax = plt.subplots()

img = mat['img']
img = img.astype(np.uint8)

ax.imshow(img)
plt.axis('off')
rect = patches.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(), linewidth=2, alpha=0.7, edgecolor='red', facecolor='none')
ax.add_patch(rect)


plt.savefig('out.png')
plt.show()