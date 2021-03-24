import cv2

img  = cv2.imread('001-3-1.jpg')
img_graysacle = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
print (img)
#print img_graysacle
import cv2
import numpy as np
 
img = cv2.imread('001-3-1.jpg',cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape
 
sobel_horizontal = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 5)
sobel_vertical = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
 
cv2.imshow('Original',img)
cv2.imshow('Sobel Horizontal Filter',sobel_horizontal)
cv2.imshow('Sobel Vertical Filter',sobel_vertical)
 
cv2.waitKey(0)
import cv2
import numpy as np
 
img = cv2.imread('001-3-1.jpg',cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape
 
denoised = cv2.GaussianBlur(img,(5,5),0)
filter = cv2.Laplacian(denoised,cv2.CV_64F)
 
cv2.imshow('Original',img)
cv2.imshow('Laplacian Filter',filter)
 
cv2.waitKey(0)
import cv2
import numpy as np
 
img = cv2.imread('001-3-1.jpg',cv2.IMREAD_GRAYSCALE)
 
filter = cv2.Canny(img,100,200)
 
cv2.imshow('Original',img)
cv2.imshow('Laplacian Filter',filter)
 
cv2.waitKey(0)
from transform import four_point_transform
import numpy as np
import argparse
import cv2
 
cap = cv2.VideoCapture(0)
 
while(1):
    ret, frame = cap.read()
    gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original',frame)
    edged_frame = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edged_frame)
    k= cv2.waitKey(5)&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
