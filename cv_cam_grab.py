# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:24:54 2020

@author: benca
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
size=(244,244)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
    image_array = np.asarray(resized)

    # Display the resulting frame
    cv2.imshow('frame',resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
