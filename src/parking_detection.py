# -*- coding: utf-8 -*-
"""
Created on Mon Jan 7 2018
@author: Evan Santa
@inspiredBy: https://github.com/eladj/detectParking
"""

import yaml
from pathlib import Path
import numpy as np
import cv2 as cv

video_path = str(Path("../assets/sample_1.mp4").resolve())
yaml_parking_lots_path = str(Path("../assets/CUHKSquare.yml").resolve())

# Set capture device or file
cap = cv.VideoCapture(video_path)

# Read YAML data (parking space polygons)
with open(yaml_parking_lots_path) as stream:
    parking_lots = yaml.load(stream)

parking_lot_bounding_rects = []
parking_mask = []
parking_status = [False]*len(parking_lots)

for lot in parking_lots:
    points = np.array(lot['points'])
    rect = cv.boundingRect(points)
    parking_lot_bounding_rects.append(rect)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        print("Capture Error")
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_blur = cv.GaussianBlur(frame_gray, (5,5), 3)

    # Parking detection
    for index, lot in enumerate(parking_lots):
        rect = parking_lot_bounding_rects[index]
        # cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255))
        roi_gray = frame_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation
        cv.imshow('ROI', roi_gray)
        # laplacian = cv.Laplacian(roi_gray, cv.CV_64F)
        # cv.imshow('Laplacian', laplacian)
        canny = cv.Canny(roi_gray, 5, 50)
        cv.imshow("Canny", canny)
        cv.waitKey(1)

        mean = np.mean(np.abs(canny))
        occupied = mean > 20
        parking_status[index] = occupied

    # Parking overlay
    for index, lot in enumerate(parking_lots):
        points = np.array(lot['points'])
        if parking_status[index]: color = (0,0,255)
        else: color = (0,255,0)
        cv.drawContours(frame, [points], contourIdx=-1, color=color, thickness=2, lineType=cv.LINE_8)

    # Display video
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
