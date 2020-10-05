# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:54:36 2020

@author: benca
"""

import tensorflow.keras
import numpy as np
import time
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('hand_phone_keras_model.h5', compile = True)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Webcam
cap = cv2.VideoCapture(0)

# The confidence for the result to be displayed
confidence = 0.92

while(True):

    # capture frame
    ret, frame = cap.read()

    # downsample image
    size = (224, 224)
    resized = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)

    # Normalize the image
    normalized_image_array = (resized.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    prediction = model.predict(data)
    
    
    print("---")
    if (prediction[0,0] > confidence) or (prediction[0,1] > confidence):
    
        if (prediction[0,0] > prediction[0,1]):
            print("PHONE")
        else:
            print("HAND")
       
    # Display the resulting frame
    cv2.imshow('frame',resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()        


