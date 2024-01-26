import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('229-video.mp4')
# ip cam addres http://192.168.217.103/mjpg/video.mjpg
backSub = cv2.createBackgroundSubtractorMOG2()

while True:

    ret, frame = cap.read()

    frame = cv2.resize(frame,(640,480))
    frame_out = frame.copy()
    imgNoBg = backSub.apply(frame)



    contours, hierarchy = cv2.findContours(imgNoBg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_ct = cv2.drawContours(frame, contours, -1, (0,255,0), 2)



    retval, mask_thresh = cv2.threshold(imgNoBg, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    countours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for countour in countours:
        if cv2.contourArea(countour) < 200:
            continue
        (x, y, w ,h) = cv2.boundingRect(countour)
        cv2.rectangle(frame_out, (x,y), (x+w,y+h), (0, 0, 255), 2)



    cv2.imshow('final', frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



