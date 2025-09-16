# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


# Lowe's ratio algorithm used in FLANN (https://www.geeksforgeeks.org/python-opencv-drawmatchesknn-function/)
def match_descriptors_lowe(desc1, desc2, ratio_thresh, dist_thresh):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        nearest = np.argsort(distances)[:2]

        # Lowe's ratio test
        if distances[nearest[0]] < ratio_thresh * distances[nearest[1]]:
            best_dist = distances[nearest[0]]
            if best_dist < dist_thresh:
                matches.append((i, nearest[0]))

    return matches

def compute_sift_keypoints_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def compute_homography_from_matches(kp1, kp2, matches):
    pts1 = np.float32([kp1[i].pt for i, _ in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[j].pt for _, j in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)  # Homography matrix to warp img2 -> img1
    return H

def blend_images(stitched, warped_img2):
    # Convert to float for safe math
    stitched_f = stitched.astype(np.float32)
    warped_f = warped_img2.astype(np.float32)

    # Create masks
    mask1 = np.any(stitched_f > 0, axis=2, keepdims=True)
    mask2 = np.any(warped_f > 0, axis=2, keepdims=True)

    # Average where both have content
    blended = np.where(mask1 & mask2,
                       (stitched_f + warped_f) / 2,
                       np.where(mask2, warped_f, stitched_f))

    return blended.astype(np.uint8)

def compute_overlap_matrix(imgs, match_threshold):
    N = len(imgs)
    overlap_matrix = np.zeros((N, N), dtype=np.uint8)
    np.fill_diagonal(overlap_matrix, 1)  # self-overlap

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # skip self-comparison

            # SIFT keypoints + descriptors
            kp1, d1 = compute_sift_keypoints_descriptors(imgs[i])
            kp2, d2 = compute_sift_keypoints_descriptors(imgs[j])
            matches = match_descriptors_lowe(d1, d2, 0.85, 100)

            if len(matches) > match_threshold:
                overlap_matrix[i][j] = 1

    return overlap_matrix


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    overlap_matrix= compute_overlap_matrix(imgs, 30)

    PADDING = 600 # Make sure the final panorama is big enough

    # Manually offset img1 into padded space
    stitched = cv2.copyMakeBorder(imgs[0], PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, (0, 0, 0))
    del imgs[0] # Remove the image from the pool after stitching

    # Start stitching loop
    while True:
        stitched_kp, stitched_desc = compute_sift_keypoints_descriptors(stitched)
        stitched_something = False

        for i, img in enumerate(imgs):
            kp, desc = compute_sift_keypoints_descriptors(img)
            matches = match_descriptors_lowe(stitched_desc, desc, 0.85, 100)

            if len(matches) >= 30: # Determine a good matche to have at least 30 matching pairs
                H = compute_homography_from_matches(stitched_kp, kp, matches)
                if H is not None:
                    warped = cv2.warpPerspective(img, H, stitched.shape[1::-1])
                    stitched = blend_images(stitched, warped)
                    del imgs[i] # Remove the image from the pool after stitching
                    stitched_something = True

        # No more compatible image to stitch
        if not stitched_something:
            break
    
    cv2.imwrite(savepath, stitched)
    return overlap_matrix
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
