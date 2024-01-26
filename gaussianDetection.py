import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture('232-video.mp4')


# Set the history and threshold for the background subtraction algorithm
history = 500

motion_threshold = 50

_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25,25),0)
previous_frame = gray

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25,25),0)

    frame_diff = cv2.absdiff(previous_frame,gray)

    _,threshold = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

    countours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countour in countours:
        if cv2.contourArea(countour) < 500:
            continue
        (x, y, w ,h) = cv2.boundingRect(countour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Detection', 700, 700)
    cv2.imshow('Motion Detection', frame)

    # Display the foreground mask
    cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gray', 700, 700)
    cv2.imshow('Gray', threshold)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()