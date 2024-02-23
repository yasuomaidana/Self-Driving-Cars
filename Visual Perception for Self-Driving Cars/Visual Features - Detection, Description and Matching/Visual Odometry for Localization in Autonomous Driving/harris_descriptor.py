import cv2
import numpy as np
from matplotlib import pyplot as plt

from m2bk import DatasetHandler


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


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def extract_features_harris_corner(image):
    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        image_gray = image

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


def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images

    """
    kp_list = []
    des_list = []

    for img in images:
        kps, desc = extract_features_function(img)
        kp_list.append(kps)
        des_list.append(desc)
    # kp_list, des_list = zip(*[extract_features_function(img) for img in images])

    return kp_list, des_list


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    matcher = cv2.BFMatcher()
    match = matcher.knnMatch(des1, des2, k=2)

    return match


def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """

    return sorted([m for m, n in match if m.distance < (dist_threshold * n.distance)], key=lambda x: x.distance)


dataset_handler = DatasetHandler()  # Load data

images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features_harris_corner)

# Visualize n first matches, set n to None to view all matches
n = 20
filtering = True

i = 0
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i + 1]

kp1 = kp_list[i]
kp2 = kp_list[i + 1]

des1 = des_list[i]
des2 = des_list[i + 1]

match = match_features(des1, des2)
if filtering:
    dist_threshold = 0.6
    match = filter_matches_distance(match, dist_threshold)

visualize_matches(image1, kp1, image2, kp2, match[:n])
plt.show()
