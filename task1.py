import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture('http://192.168.217.103/mjpg/video.mjpg')

# Set the history and threshold for the background subtraction algorithm
history = 500

threshold = 50


# Create the background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold)

# Define the kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break  # break the loop if we have reached the end of the video

    # Apply the background subtraction
    fgMask = backSub.apply(frame)

    # Apply morphological operations to remove noise and fill in holes
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original frame
    for contour in contours:
        # You can set a threshold to detect only significant motion
        if cv2.contourArea(contour) > 60:
            # Get the bounding box coordinates from the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the detected motion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the original frame with the contours superimposed
    cv2.imshow('Motion Detection', frame)

    # Display the foreground mask
    cv2.imshow('Foreground Mask', fgMask)

    # Wait for the user to press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
