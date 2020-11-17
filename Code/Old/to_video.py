# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:53:17 2020

@author: Yann
"""

import cv2
import glob

def to_video(path):
    img_array = []
    for filename in glob.glob(path + '*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    
    out = cv2.VideoWriter(path+'model.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
