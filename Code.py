import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("Sample Video.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

  
sub = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
ratio = 1.0
image = cv2.resize(frame, (0, 0), None, ratio, ratio)
width2, height2, channels = image.shape

while True:
    ret, frame = cap.read()
    if not ret:
        frame = cv2.VideoCapture("Sample Video.mp4")
        continue
    if ret:
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)
        cv2.imshow("image", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        fgmask = sub.apply(gray)
        cv2.imshow("fgmask", fgmask) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("closing", closing)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        cv2.imshow("opening", opening)
        dilation = cv2.dilate(opening, kernel)
        cv2.imshow("dilation", dilation)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
        cv2.imshow("retvalbin", retvalbin)
        im2, contours, hierarchy = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 400
        maxarea = 50000
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))
        
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])
                if minarea < area < maxarea:
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
    cv2.imshow("countours", image)
    key = cv2.waitKey(20)
    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()