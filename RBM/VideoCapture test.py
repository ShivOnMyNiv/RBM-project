# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:10:38 2020

@author: derph
"""
    
import cv2
import numpy as np
from scipy import signal as sg
from PIL import Image
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)

vedge_detector = np.array([[[ -1, 0, 1],
                          [ -2, 0, 2],
                          [ -1, 0, 1]]]) 
hedge_detector = np.array([[[ -1, -2, 1],
                          [ 0, 0, 0],
                          [ 1, 2, 1]]]) 

pic = None
vedge = None
hedge = None

# Reading photos
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame    
    cv2.imshow('gray frame', gray[0:,0:])
    cv2.imshow('frame', frame[0:,0:])
    if cv2.waitKey(100) & 0xFF == ord(' '):
        pic = frame
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        print(pic.shape)
        vedge = sg.convolve(frame, vedge_detector, mode='same')
        hedge = sg.convolve(frame, hedge_detector, mode='same')
        break
cv2.destroyAllWindows()

# Saving photo
cv2.imshow("picture", pic)
Image.fromarray(pic).save("Selfie.png")


# Pulling photo
im = cv2.imread('Selfie.png')
plt.imshow(hedge)
cv2.imshow('Reading Photo', im)
cv2.waitKey(0) 

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
del cam