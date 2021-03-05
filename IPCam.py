# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:52:06 2020

@author: BiO
"""

import cv2

url = "http://10.13.35.151:4747/video"
video = cv2.VideoCapture(url)

while True:
    ret, frame = video.read()
    if ret == True:
        cv2.imshow("IPCam",frame)
    if cv2.waitKey(1) == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
                