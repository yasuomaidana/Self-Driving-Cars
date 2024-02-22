import cv2
import numpy as np
from matplotlib import pyplot as plt


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image)


def extract_features_harris_corner(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # corner harris
    dst = cv2.cornerHarris(image_gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    threshold, key_points = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    key_points = np.argwhere(key_points > threshold)
    key_points = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in key_points]

    # # refining keypoints
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners_refined = cv2.cornerSubPix(image_gray, keypoints, (5, 5), (-1, -1), criteria)

    # Keypoints and descriptors
    sift = cv2.SIFT.create()  # Or cv2.xfeatures2d.SURF_create()
    kp, des = sift.compute(image_gray, key_points)  # Or surf.compute(img_gray, corners_refined)
    return kp, des


src = cv2.imread(cv2.samples.findFile("apple.png"))
cv2.imshow("original", src)
key, dest = extract_features_harris_corner(src)
display_ = cv2.drawKeypoints(src, key, None)
cv2.imshow("featured", display_)
visualize_features(src, key)
plt.show()
cv2.waitKey(0)
