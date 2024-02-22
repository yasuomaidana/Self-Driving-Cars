import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from m2bk import DatasetHandler

dataset_handler = DatasetHandler()


img = dataset_handler.images[0]
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(img, cmap='gray')
plt.show()


# Initiate FAST object with default values
fast = cv.FastFeatureDetector.create()
# find and draw the key-points
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
cv.imwrite('fast_true.png', img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imwrite('fast_false.png', img3)