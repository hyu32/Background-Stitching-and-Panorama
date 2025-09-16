# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Select stictly brighter color pixel to remove foreground of people wearing dark clothing
def merge_foreground(overlapimg, img1):
    # Convert to float32 for pixel math
    overlapimg = overlapimg.astype(np.float32)
    img1 = img1.astype(np.float32)

    # Compute per-pixel intensity sum
    clipped_sum = np.sum(overlapimg, axis=2, keepdims=True)
    img1_sum = np.sum(img1, axis=2, keepdims=True)

    # Only use overlapimg where it's strictly brighter than img1
    use_clipped = (clipped_sum > img1_sum)

    # Apply merging logic
    new = np.where(use_clipped, overlapimg, img1)

    return new.astype(np.uint8)

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    # Pad img1 to ensure img1 is larger than img2
    img1 = cv2.copyMakeBorder(img1, 150, 150, 150, 150, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Conver img to gray for easier feature extracting
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT (Scale-Invariant Feature Transform) to extract features (https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Feature matching
    matches = match_descriptors_lowe(des1, des2, 0.85, 100) # ratio_thresh 0.85, dist_thresh 100

    pts1 = np.float32([kp1[i].pt for i, _ in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[j].pt for _, j in matches]).reshape(-1, 1, 2)

    # Warp img2 -> img1 using homography matrix derived from findHomography()
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # Extract overlap region
    overlapimg = warped[0:img1.shape[0], 0:img1.shape[1]]

    merged = merge_foreground(overlapimg, img1)

    cv2.imwrite(savepath, merged)
    return

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

