import cv2
import numpy as np

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Function to get good features to track
def get_good_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)


# Initialize the camera and grab a reference to the first frame
capture = cv2.VideoCapture('http://192.168.217.103/mjpg/video.mjpg')  # Device index (0 for default camera)

# Take first frame and find corners in it
ret, old_frame = capture.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = get_good_features(old_frame)  # Points to track

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while capture.isOpened():
    ret, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # if p1 is None, simply continue without drawing lines
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to int
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
        img = cv2.add(frame, mask)
    else:
        print("No points to track.")

    # Show the frame with optical flow tracks
    cv2.imshow('Optical Flow - Lucas-Kanade', img)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
