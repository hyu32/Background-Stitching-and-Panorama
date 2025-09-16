University at Buﬀalo
Department of Computer Science and Engineering

Task 1 Background Stitching
The goal of this task is to experiment with image stitching methods. Two images may have the
same background but diﬀerent foreground. For example, a moving person may be moving in the
scene. You need to stitch the two images into one image eliminating foreground objects that move
in the scene. You may assume that the region where a foreground object covers the background
has similar characteristics to the rest of the background and is visible in two or more images.
The shape of the output image after stitching might be irregular since you
cannot crop either of the transformed images.
Steps:
• Extract a set of key points for each image.
• Extract features from each key point.
• Match features and use matches to determine if there is overlap between given pairs of images.
• Compute the homography between the overlapping pairs as needed. Use RANSAC to optimize
your result.
• Transform the images and stitch the two images into one mosaic, eliminating the foreground
as described above, but do NOT crop your image.
• Save the resulting mosaic according to the instructions specified in the script.

Task 2 Image Panorama
This task aims to stitch multiple images into one panoramic photo. The given
images might be non-overlapping or multiply-overlapped (overlapping two or more other images).
The shape of the output image after stitching might be irregular since you cannot crop either of
the transformed images. There are no restrictions regarding the method you use to stitch photos
into a panoramic photo. You can assume the following:
• Your code will need to be able to stitch together four or more images, and you will not know
that in advance.
• You can assume that IF an image is to be part of the panorama, it will overlap at least one
other image by at least 20%.
• Images that do not overlap with any other image can be ignored.
• Images can overlap with multiple images.
• Although the Figure below shows horizontal panoramas, your five images can be stitched
together in any way.
• You are only expected to produce one overall image.
• While some of the most modern techniques may use a spherical projection for better panora-
mas, you are free to assume that basic 2D planer transformations are suﬃcient for this project.
Steps:
Page 2 of 5
CSE 473/573 Project #2
• Extract features for each image, match features, and determine the spatial overlaps of the
images automatically. For instance, among four images, if image1, image3 and image4 overlap
with each other, then your overlap array should be
   
1 0 1 1
0 1 0 0
1 0 1 1
1 0 1 1
   . Follow the instructions in
the provided script and save the overlapping result as a N ∗ N one-hot array in the JSON file.
• Conduct image transformation and stitch into one panoramic photo. Save the photo as the
instructions specified in the script.

Task 3 Image Panorama Example
Find something that is Buﬀalo or UB-related (suggested by not required), photograph it, and
demonstrate that your code works on regions where at least four images overlap IN A VERTICAL
DIRECTION. Set the number of your images N , and name your input images as specified in the
script.
CSE 473/573 Project #2
OpenCV(3.4.2.17) has been tested and it does provide a fully functional API like
cv2 . xfeatures2d . SIFT create ( )
Note: You may need administrative privilege to execute the installation on your machine.
Page 5 of 5
