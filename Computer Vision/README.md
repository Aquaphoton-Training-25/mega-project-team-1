## 3D stereo vision 

code:
```
import cv2
import numpy as np

# Load images
img1 = cv2.imread("new folder/3/im0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("new folder/3/im1.png", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
#detecting all the different  features in our image
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches for a specific range
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[300:600], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_resized=cv2.resize(img_matches, (700, 700))
# Display the matches
cv2.imshow("Matches", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Extract matched keypoints' coordinates  (list or array of points )

pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate the Fundamental matrix using the RANSAC method
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)


# Select inlier points based on the mask
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

cam0=np.array([[5806.559, 0, 1429.219],[0, 5806.559, 993.403],[0, 0, 1]])
cam1=np.array([[5806.559, 0, 1543.51],[0,5806.559,993.403],[0, 0, 0]])
K1 = cam0
K2 = cam1

# Calculate the Essential matrix E
E = K2.T @ F @ K1

# Decompose the Essential matrix into rotation and translation
_, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K1)

# Print results
print("Fundamental Matrix (F):\n", F)
print("\nEssential Matrix (E):\n", E)
print("\nRotation Matrix (R):\n", R)
print("\nTranslation Vector (t):\n", t)


```
![image](https://github.com/user-attachments/assets/2c5aaea0-cf47-4612-a9b2-4bf700a5a920)
![image](https://github.com/user-attachments/assets/21c46235-3a68-45cf-8527-d3377c86f386)

### the final code 
```
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load images for calibration
img1 = cv2.imread("new folder/2/im0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("new folder/2/im1.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blurring to reduce noise
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

# Initialize SIFT detector and find keypoints and descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints' coordinates
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate Fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

# Camera intrinsic matrices
K1 = np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]])
K2 = np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
baseline = 144.049  # in mm
doffs = 185.788     # in mm
width=2880
height=1980

# Decompose the Essential matrix into rotation and translation
E = K2.T @ F @ K1
_, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K1)
print("Fundamental Matrix (F):\n", F)
print("\nEssential Matrix (E):\n", E)
print("\nRotation Matrix (R):\n", R)
print("\nTranslation Vector (t):\n", t)
# Rectify images
retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_inliers, pts2_inliers, F,  (width, height))
print("Homography for Image 1:\n", H1)
print("Homography for Image 2:\n", H2)

img1_rectified = cv2.warpPerspective(img1, H1, img1.shape[1::-1])
img2_rectified = cv2.warpPerspective(img2, H2, img2.shape[1::-1])

# Plot epipolar lines
def plot_epipolar_lines(img1, img2, lines, pts1, pts2):
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
    return img1, img2
#This step calculates these lines to visualize the accuracy of the rectification and the corresponding points.
lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

img1_lines, img2_lines = plot_epipolar_lines(img1.copy(), img2.copy(), lines1, pts1_inliers, pts2_inliers)

# Display rectified images with epipolar lines
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img1_lines, cmap='gray')
plt.title('Image 1 with Epipolar Lines'), plt.axis('off')
plt.subplot(122), plt.imshow(img2_lines, cmap='gray')
plt.title('Image 2 with Epipolar Lines'), plt.axis('off')
plt.show()

# Disparity map calculation using StereoSGBM
min_disp = 0
num_disp = 640
block_size = 11  # Matching block size, must be odd

# StereoSGBM setup
# Various parameters are tuned for better accuracy.
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,  # Regularization parameter
    P2=32 * 3 * block_size ** 2,  # Continuity constraint
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

# Normalize and save the disparity map
disparity_rescaled = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_rescaled = np.uint8(disparity_rescaled)

cv2.imwrite("disparity_grayscale.png", disparity_rescaled)

# Convert the disparity map to a color image
disparity_color = cv2.applyColorMap(disparity_rescaled, cv2.COLORMAP_JET)
cv2.imwrite("disparity_color.png", disparity_color)

# Display disparity maps
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(disparity_rescaled, cmap='gray')
plt.title('Disparity Map (Grayscale)'), plt.axis('off')
plt.subplot(122), plt.imshow(disparity_color)
plt.title('Disparity Map (Color)'), plt.axis('off')
plt.show()

# Calculate depth map
focal_length = K1[0, 0]  # in pixels
#this gives the distance from the camera to the object in the scene.
depth = (focal_length * baseline) / (disparity_rescaled.astype(np.float32) + 1e-6)

# Normalize and save depth images
depth_rescaled = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_rescaled, cv2.COLORMAP_JET)

cv2.imwrite("depth_gray.png", depth_rescaled)
cv2.imwrite("depth_color.png", depth_color)

# Show depth images
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(depth_color)
plt.title('Depth Image - Color'), plt.axis('off')
plt.subplot(122), plt.imshow(depth_rescaled, cmap='gray')
plt.title('Depth Image - Grayscale'), plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/a690c101-20bc-4b6e-9e56-5464c89f830b)
![image](https://github.com/user-attachments/assets/c861cf6c-9e18-4728-a8da-5f624235b41e)
![image](https://github.com/user-attachments/assets/58625416-4df5-45ac-ae47-5249d7c60299)


