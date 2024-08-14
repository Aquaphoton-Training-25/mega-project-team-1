## 3D stereo vision 

code:
```
import cv2

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

```
