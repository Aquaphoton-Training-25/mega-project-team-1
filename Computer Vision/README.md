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

