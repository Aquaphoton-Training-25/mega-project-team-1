import cv2
import numpy as np
import imutils

# Open the left and right video captures
cap_left = cv2.VideoCapture("Left (Better Quality).mp4")
cap_right = cv2.VideoCapture("Right(Better Quality).mp4")

# Get FPS and calculate the delay for each frame
fps = cap_left.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Check if both videos were opened successfully
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error opening video streams")
    exit()

# Create the stitcher object
image_stitcher = cv2.Stitcher.create()

# Define the desired size for the frames (e.g., 640x480 for lower resolution)
desired_width = 640
desired_height = 480
desired_size = (desired_width, desired_height)

# Define a fixed size for all stitched images (e.g., 1280x720)
fixed_size = (1280, 720)

# Loop through frames
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    # Break the loop if either video reaches the end
    if not ret_left or not ret_right:
        break

    # Resize both frames to the desired size before stitching
    frame_left_resized = cv2.resize(frame_left, desired_size)
    frame_right_resized = cv2.resize(frame_right, desired_size)

    # Store the resized frames in a list
    frames = [frame_left_resized, frame_right_resized]

    # Perform stitching
    error, stitched_image = image_stitcher.stitch(frames)

    # If stitching is successful, display the stitched image
    if error == cv2.Stitcher_OK:
        
        #some image processing to refine the output
        stitched_image = cv2.copyMakeBorder(stitched_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0,0))

        gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)

        cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)

        min_rectangle = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            min_rectangle = cv2.erode(min_rectangle, None)
            sub = cv2.subtract(min_rectangle, thresh_img)

        contours = cv2.findContours(min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(areaOI)

        stitched_image = stitched_image[y:y + h, x:x + w]

        # Resize the stitched image to a fixed size
        stitched_image = cv2.resize(stitched_image, fixed_size)

        # Display the stitched image directly (no further resizing needed)
        cv2.imshow('stitched image', stitched_image)
    else:
        print(f"Stitching failed with error code: {error}")
        continue

    # Exit when 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video captures and close windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
