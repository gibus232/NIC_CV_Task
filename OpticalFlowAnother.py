import numpy as np
import cv2
import time

#Задание параметров для алгоритма

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

cap = cv2.VideoCapture('232-video.mp4')
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25,25),0)
previous_frame = gray

motion_threshold = 20
while True:

    # start time to calculate FPS
    start = time.time()

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25,25),0)

    frame_diff = cv2.absdiff(previous_frame,gray)

    _,threshold = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

    countours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countour in countours:
        if cv2.contourArea(countour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(countour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = frame.copy()
    if len(trajectories) > 0:
        img0, img1 = prev_gray, gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = gray

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Optical Flow', 700, 700)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 700,700)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()