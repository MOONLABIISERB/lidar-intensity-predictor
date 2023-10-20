#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:45:24 2022

@author: dse
"""
import cv2
import glob

lst = glob.glob(r'./intensity_pred/*_g.png')
out = cv2.VideoWriter('./intensity_gt.avi',cv2.VideoWriter_fourcc(*"XVID"), 4, (2400,2400))
for i in range(1000):
    if i%2 ==0:
        continue
    else:
        img = cv2.imread('./intensity_pred/'+str(i)+'_g.png')
    out.write(img)
out.release
    