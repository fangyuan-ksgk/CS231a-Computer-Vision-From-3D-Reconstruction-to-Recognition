I built all functions (including utils) in my own ways and include the solution & derivation in this folder.

Assignment 2:

1. Fundamental Matrix Estimation From Point Correspondences (8-point algorithm & normalized 8-point algorithm), fundamental matrix can be computed with 8 corresponding points from two images obtained by two different camera (intrinsic & extrinsic matrix differs). We can use Fundamental Matrix to plot the Epipolar lines on both images, which can be used for image rectification.

![image](https://user-images.githubusercontent.com/66006349/150302663-84ee8e6d-6818-48a5-a496-685905f389e0.png)

![image](https://user-images.githubusercontent.com/66006349/150302707-97aaae42-5a8f-4ac4-adc9-1f01dc2854f4.png)

Accuracy of the estimated Fundamental matrix can be evaluated by the average distance between points and their corresponding epipolar lines (in both images from two camera systems).

2. Image Rectification with matching homographies.

![image](https://user-images.githubusercontent.com/66006349/150671052-1595f17a-17ea-4d55-9cd7-5ba6e0b79f6d.png)

Rectified images pair are very useful for various maneuver including stereo and etc. Corresponding epipolar lines are horizontal and alinged in the two images from two camera.

3. Affine Structure from Motion (SFM) problem: factorization method (Kanade method).

![image](https://user-images.githubusercontent.com/66006349/154840029-0300b8df-c3fa-461f-90ab-1f8e75291edc.png)

From multiple views, retrieve the 3d structure under affine camera assumptions.

4. General Structure From Motion problem with calibrated cameras:

![image](https://user-images.githubusercontent.com/66006349/154840114-2a34133b-6087-4181-8cf2-2c2ce95c8fde.png)

Given four images of a statue token from four calibrated cameras at different viewpoint, estimate the structure with bundle adjustment.



