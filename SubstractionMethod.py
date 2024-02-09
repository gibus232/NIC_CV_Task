import math
from collections import deque
from random import random

import cv2
import numpy as np
import time

#Задание параметров для алгоритма Лукаса-Канаде и good features to track

lk_params = dict(winSize=(15, 15),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=30,
                      qualityLevel=0.1,
                      minDistance=10,
                      blockSize=5)

trajectory_len = 400
detect_interval = 2
trajectories = []
frame_idx = 0
points = []
cap = cv2.VideoCapture('232-video.mp4')



blank = np.zeros((480,640,3),dtype=np.uint8)

# ip cam addres http://192.168.217.103/mjpg/video.mjpg
backSub = cv2.createBackgroundSubtractorMOG2()
center_pointsX = 0
center_pointsY = 0
cX = 0
cY = 0
ocx = 0
ocy = 0
pointsDict = {'cnt num':'points'}
while True:


    start = time.time()

    _, frame = cap.read()

    frame = cv2.resize(frame,(640,480))

    frame_out = frame.copy()
    imgNoBg = backSub.apply(frame)

    _, mask_thresh = cv2.threshold(imgNoBg, 150, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel2)

    mask_eroded = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, kernel)

    mask_eroded = cv2.dilate(mask_eroded, None, iterations=2)
    countours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = frame_out.copy()
    for countour in countours:
        if cv2.contourArea(countour) < 6500:
            continue
        maxc = max(countours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(maxc)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
        cX = int(x+w / 2)
        cY = int(y+h / 2)
        center = (cX,cY)
        cv2.circle(img, (cX, cY), 5, (0, 255, 255), -1)

    if ocx !=0 and ocx:
        cv2.line(blank, (ocx, ocy), (cX, cY), (255, 255, 255), 5)
    img = cv2.add(blank,img)
    ocx = cX
    ocy = cY

    center_pointsX = []
    center_pointsY = []

    # Расчет оптического потока
    if len(trajectories) > 0:
        img0, img1 = prev_gray, mask_eroded
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)

        p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Building trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Last points detetcted
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Drawing trajectories lines
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Update interval
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(mask_eroded)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        goodF = cv2.goodFeaturesToTrack(mask_eroded, mask=mask, **feature_params)
        if goodF is not None:
            for x, y in np.float32(goodF).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = mask_eroded

    end = time.time()

    fps = 1 / (end - start)
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('final', img)
    cv2.imshow('mask', mask)
    cv2.imshow('maskeroded', mask_eroded)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



