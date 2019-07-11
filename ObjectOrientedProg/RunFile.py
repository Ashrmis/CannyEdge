import CannyEdgeDetector_OOP as CED
import cv2
from scipy import signal
from matplotlib import pyplot as plt
img = cv2.imread(r'C:\Users\Ashish\Desktop\Graduate School\Study\images\lena.png')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayImage=cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)

# image,size,sigma,weakpix,strongpix,lowthresh,highthresh)
detector=CED.cannyedge(grayImage,5,2,100,255,.05,.17)

finalIMAGE=detector.detect()
plt.imshow(finalIMAGE,cmap = 'gray')
plt.show()