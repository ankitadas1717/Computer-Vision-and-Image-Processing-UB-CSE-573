"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random
import math

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    'gray scale image conversion'
    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)  # left-image in gray scale
    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)  # left-image in gray scale

    'SIFT operation'
    sift = cv2.xfeatures2d.SIFT_create()  # applying sift
    left_img_keypoint, left_img_descriptor = sift.detectAndCompute(left_img_gray, None)
    right_img_keypoint, right_img_descriptor = sift.detectAndCompute(right_img_gray, None)

    'FLANN matching technique to find K Nearest Neighbours'
    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    'Creating object for FLANN matching'
    doFlann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = doFlann.knnMatch(left_img_descriptor, right_img_descriptor, k=2)

    Keypoints = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            Keypoints.append(m)

    MIN_MATCHES = 10

    if len(Keypoints) > MIN_MATCHES:
        coordinate1 = np.array([left_img_keypoint[m.queryIdx].pt for m in Keypoints])
        np.reshape(coordinate1, (-1, 1, 2))
        coordinate1 = np.float32(coordinate1)
        coordinate2 = np.array([right_img_keypoint[m.trainIdx].pt for m in Keypoints])
        np.reshape(coordinate2, (-1, 1, 2))
        coordinate2 = np.float32(coordinate2)

        H, homographyStatus = cv2.findHomography(coordinate1, coordinate2, cv2.RANSAC, 5.0)
    else:
        print(" Enough matches have not been found - %d/%d" % (len(Keypoints), MIN_MATCHES))

    'Shapes of the two images'
    left_height, left_width = left_img_gray.shape
    right_height, right_width = right_img_gray.shape

    'Corners of each image'
    points_m1 = np.float32([[0, 0], [0, left_height], [left_width, left_height], [left_width, 0]]).reshape(-1, 1, 2)
    points_m2 = np.float32([[0, 0], [0, right_height], [right_width, right_height], [right_width, 0]]).reshape(-1, 1, 2)

    'Translation and rotation of each corner of left image with respect to the right image using Homography matrix'
    points_m1_modify = cv2.perspectiveTransform(points_m1, H)

    '''Add the translated corners of the left image to the corners of the right image in the form of a 
        list of list for each corner location on the canvas. We have added along axis=0, i.e. along the rows'''
    all_mPoints = np.concatenate((points_m2, points_m1_modify), axis=0)

    '''Finding the minimum and maximum values from all the corners to create the size of the frame at a 
     distance of +-1 '''
    [pano_xmin, pano_ymin] = np.int32(all_mPoints.min(axis=0).ravel() - 1.0)
    [pano_xmax, pano_ymax] = np.int32(all_mPoints.max(axis=0).ravel() + 1.0)

    'Calculation of the left upper most and lower most corners of the frame'
    transformationM = [-pano_ymin, -pano_xmin]

    'Compute the Translated matrix for the new frame, with respect to which the H will be translated'
    translatedH = np.array([[1, 0, transformationM[1]], [0, 1, transformationM[0]], [0, 0, 1]])

    'Warp the left image with respect to the right image'
    img_panorama = cv2.warpPerspective(left_img, translatedH.dot(H), (pano_xmax - pano_xmin, pano_ymax - pano_ymin))
    img_panorama[transformationM[0]:right_height + transformationM[0],
    transformationM[1]:right_width + transformationM[1]] = right_img

    return img_panorama

    #raise NotImplementedError

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


