import numpy as np
import cv2

cap_video = cv2.VideoCapture("227-video.mp4")

feature_parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

lk_parameters = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

random_color = np.random.randint(0, 255, (100, 3))

ret, previous_frame = cap_video.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
p0_point = cv2.goodFeaturesToTrack(previous_gray, mask=None, **feature_parameters)

mask_drawing = np.zeros_like(previous_frame)
while 1:
    ret, frame = cap_video.read()
    if not ret:
        print("No frames available!")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st_d, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, frame_gray, p0_point, None, **lk_parameters
    )

    if p1 is not None:
        good_new_point = p1[st_d == 1]
        good_old_point = p0_point[st_d == 1]

    for i, (new_point, old_point) in enumerate(zip(good_new_point, good_old_point)):
        a, b = new_point.ravel()
        c, d = old_point.ravel()
        mask_drawing = cv2.line(
            mask_drawing,
            (int(a), int(b)),
            (int(c), int(d)),
            random_color[i].tolist(),
            2,
        )
        frame = cv2.circle(frame, (int(a), int(b)), 5, random_color[i].tolist(), -1)
    img = cv2.add(frame, mask_drawing)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 700, 700)
    cv2.imshow("frame", img)

    previous_gray = frame_gray.copy()
    p0_point = good_new_point.reshape(-1, 1, 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap_video.release()
cv2.destroyAllWindows()