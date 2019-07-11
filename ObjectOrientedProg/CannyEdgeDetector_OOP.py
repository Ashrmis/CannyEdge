# Canny Edge Operator Object Oriented Programming
import numpy as np
from scipy import signal

class cannyedge:
    def __init__(self,image,size,sigma,weakpix,strongpix,lowthresh,highthresh):
        self.imgs=image
        self.size=size
        self.imgs_final=[]
        self.sigma=sigma
        self.image=image
        self.weakpix=weakpix
        self.strongpix=strongpix
        self.lowthresh=lowthresh
        self.highthresh=highthresh
        
        self.img_smoothed=None
        self.gradientMat=None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        
        return
    def gaussian_kernel(self,size,sigma):
        size = size //2
        x,y=np.mgrid[-size:size+1,-size:size+1] # create a mesh grid
        cons=(1/(2*np.pi*sigma**2))
        kernel=np.exp(-((x**2+y**2)/(2*sigma**2)))*cons
        
        return kernel
    
    def sobelfilter(self,smoothed_image):
        Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
        Gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)
        
        Ix=signal.convolve2d(smoothed_image,Gx)
        Iy=signal.convolve2d(smoothed_image,Gy)
        
        magnitude=np.sqrt(Ix**2+Iy**2)
        orgmag=magnitude
        magnitude=magnitude/magnitude.max() *255
        direction=np.arctan2(Iy,Ix)
    
        return magnitude,direction;
    
    def non_maxsup(self,img, theta):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = theta * 180 / np.pi
        angle[angle < 0] += 180 
        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
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
    
    def threshold(self, img):

        highThreshold = img.max() * self.highthresh;
        lowThreshold = highThreshold * self.lowthresh;
    
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)
    
        weak = np.int32(self.weakpix)
        strong = np.int32(self.strongpix)
    
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)   
    
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
    
        return (res)
        
    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weakpix
        strong = self.strongpix
    
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
    
        return img
        
    def detect(self):
        img_final=[]
        self.img_smoothed=signal.convolve2d(self.imgs,self.gaussian_kernel(self.size,self.sigma))
        self.gradientMat,self.thetaMat=self.sobelfilter(self.img_smoothed)
        self.nonMaxImg=self.non_maxsup(self.gradientMat,self.thetaMat)
        self.thresholding=self.threshold(self.nonMaxImg)
        img_final=self.hysteresis(self.thresholding)
        self.img_final=img_final
        return self.img_final

