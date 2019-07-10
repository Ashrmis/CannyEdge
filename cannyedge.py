# Canny Edge Detector without object oriented progamming

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load image and convert to grayscale
img = cv2.imread('image1.jpg')
print(img.shape)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grayImage=cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)
print(grayImage.shape)

plt.subplot(1, 2, 1)
plt.xticks([]), plt.yticks([]) 
plt.imshow(RGB_img)

plt.subplot(1, 2, 2)
plt.xticks([]), plt.yticks([]) 
plt.imshow(grayImage,cmap = 'gray')

plt.show()

# 1. Noise Reduction with Gaussian
# kernel size is 2k+1 x 2k+1 want to have it as odd so... // 2
def gaussian_kernel(size,sigma):
    size = size //2
    x,y=np.mgrid[-size:size+1,-size:size+1] # create a mesh grid
    cons=(1/(2*np.pi*sigma**2))
    kernel=np.exp(-((x**2+y**2)/(2*sigma**2)))*cons
    return kernel

from scipy import signal
Smoothed_image=signal.convolve2d(grayImage,gaussian_kernel(5,1))

plt.subplot(1, 2, 1)
plt.xticks([]), plt.yticks([]) 
plt.imshow(grayImage,cmap = 'gray')

plt.subplot(1, 2, 2)
plt.xticks([]), plt.yticks([]) 
plt.imshow(Smoothed_image,cmap = 'gray')
plt.show()

# 2. Gradient Calculation
# sobel filter without using actual function
def sobelfilter(smoothed_image):
    Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    Gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)
    
    Ix=signal.convolve2d(smoothed_image,Gx)
    Iy=signal.convolve2d(smoothed_image,Gy)
    
    magnitude=np.sqrt(Ix**2+Iy**2)
    orgmag=magnitude
    magnitude=magnitude/magnitude.max() *255
    direction=np.arctan2(Iy,Ix)

    return magnitude,direction,orgmag;

G,theta,Gorg=sobelfilter(Smoothed_image)

plt.subplot(1, 2, 1)
plt.imshow(grayImage,cmap = 'gray')

plt.subplot(1, 2, 2)
plt.imshow(G,cmap = 'gray')
plt.show()

# 3. Non-Max Supression
def non_maxsup(img, theta):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                # if the angle at each pixel is within a range set
                # the comparison pixels equal to the corresponding line
                # pixels. If angle not within then set that non-max image
                # pixel point to zero and keep pixel comparisons at max of
                # 255.
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

supressed_image=non_maxsup(G,theta)
plt.subplot(1, 2, 1)
plt.imshow(G,cmap = 'gray')

plt.subplot(1, 2, 2)
plt.imshow(supressed_image,cmap = 'gray')
plt.show()

# 4. Thresholding 2x
def thresholding(img,lowthresh=.05,highthresh=.09):
    highThreshold=img.max()*highthresh
    lowThreshold=highThreshold*lowthresh
    
    M,N=img.shape
    res=np.zeros((M,N),dtype=np.int32)
    
    weak=np.int32(20)
    strong=np.int32(255)
    
    strong_i,strong_j=np.where(img>=highThreshold)
    zeros_i,zeros_j=np.where(img<lowThreshold)
    weak_i,weak_j=np.where((img<=highThreshold) & (img>=lowThreshold))
    res[strong_i,strong_j]=strong
    res[weak_i,weak_j]=weak
    
    return res,strong, weak;

thresholdedImage,strongloc,weakloc=thresholding(supressed_image)
plt.subplot(1, 2, 1)
plt.imshow(supressed_image,cmap = 'gray')

plt.subplot(1, 2, 2)
plt.imshow(thresholdedImage,cmap = 'gray')

plt.show()

# 5. Hysteresis
def hysteresis(img,weak,strong):
    M,N=img.shape
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                if img[i,j]==weak:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
            except IndexError as e:
                pass
    return img

hysteresis_image=hysteresis(thresholdedImage,weakloc,strongloc)
plt.subplot(1, 2, 1)
plt.imshow(thresholdedImage,cmap = 'gray')

plt.subplot(1, 2, 2)
plt.imshow(hysteresis_image,cmap = 'gray')

plt.show()
