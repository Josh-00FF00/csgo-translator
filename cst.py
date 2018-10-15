from PIL import Image, ImageOps
import pytesseract
import numpy as np
import os
import cv2

#take screenshot and convert to grayscale
#im = ImageGrab.grab(170, 670, 614, 960)

#using screenshot for now, easier to test
#crop values are the area of the screenshot where the text is (for now)
#will have to be dynamic at some point due to different resolutions, 16x9, 4x3

def contrast_brightness_adjust(img, brightness, contrast):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    return cv2.addWeighted(img, 1. + contrast/127., img, 0, brightness-contrast)

def resize_img(img, scale_percent):
    width = int(contrast.shape[1] * scale_percent / 100)
    height = int(contrast.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv2.resize(contrast, dim, interpolation = cv2.INTER_LINEAR) 

errosion_kernel = np.ones((2,1),np.uint8) #Kernel skewed high because text is usually longer than it is wide
dilation_kernel = np.ones((5,5),np.uint8)

#crop_area = ((170, 800), (614, 1000)) # (x,y) pairs, 1st top left 2nd bottom right
crop_area = ((5, 750), (414, 1000)) # (x,y) pairs, 1st top left 2nd bottom right

img = cv2.imread('img/test2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,1) 
text_area = gray[crop_area[0][1]:crop_area[1][1], crop_area[0][0]:crop_area[1][0]] #numpy slice [y:y, x:x]

thr3 = cv2.adaptiveThreshold(text_area,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,37,-135) # All values obtained via experimentation

eroded = cv2.erode(thr3,errosion_kernel,iterations = 1)
dilated = cv2.dilate(eroded,dilation_kernel,iterations = 2)

extracted = cv2.bitwise_and(text_area,text_area, mask = dilated)

inverted = cv2.bitwise_not(extracted) #swap pixel values, light becomes dark. Better for font recognition
contrast = contrast_brightness_adjust(inverted, 0, 100) #Greatly increase contrast
resized = resize_img(contrast, 300) #Tesseract likes larger images

Image.fromarray(contrast).show()

text = pytesseract.image_to_string(resized)
print(text)
