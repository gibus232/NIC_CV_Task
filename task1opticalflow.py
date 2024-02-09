
import numpy as np
import cv2

# Èíèöèàëèçàöèÿ ôîíîâîãî âû÷èòàíèÿ
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

# Ñëîâàðü äëÿ îòñëåæèâàíèÿ ïóòåé
paths = {}

# Èíèöèàëèçàöèÿ âèäåîïîòîêà
cap = cv2.VideoCapture("232-video.mp4")  # Èñïîëüçóåòñÿ êàìåðà; çàìåíèòå 0 íà 'path_to_video.mp4' äëÿ èñïîëüçîâàíèÿ âèäåîôàéëà

frame_idx = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    if not ret:
        break

    # Ôîíîâîå âû÷èòàíèå è ïîëó÷åíèå ìàñêè ïåðåäíèõ îáúåêòîâ
    fgMask = backSub.apply(frame)

    # Óäàëåíèå øóìà è íåçíà÷èòåëüíûõ îáúåêòîâ
    _, thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Ïîèñê êîíòóðîâ è îòñëåæèâàíèå îáúåêòîâ
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 250:
            continue
        # Íàõîæäåíèå öåíòðà ìàññ êîíòóðà
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Îòñëåæèâàíèå ïóòè êàæäîãî îáúåêòà
        if frame_idx not in paths:
            paths[frame_idx] = [(cx, cy)]
        else:
            paths[frame_idx].append((cx, cy))

        frame_idx += 1

    # Ðèñîâàíèå òðàåêòîðèé
    for path in paths.values():
        for i in range(1, len(path)):
            if path[i - 1] is None or path[i] is None:
                continue
            cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 2)

    # Îòîáðàæåíèå ðåçóëüòàòà
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# Î÷èñòêà ðåñóðñîâ
cap.release()
cv2.destroyAllWindows()