import CannyEdgeDetector_OOP as CED
import cv2
from scipy import signal
from matplotlib import pyplot as plt
from PIL import Image
import PIL
import os, os.path
images=[]
path= r'C:PATH_HERE'
imagetypes=['.jpg','.png','.jpeg']

for f in os.listdir(path):
    extension=os.path.splitext(f)[1]
    if extension not in imagetypes:
        continue
    images.append(Image.open(os.path.join(path,f)))



# np.asarray(images[0])
    
#RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#grayImage=cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)

# image,size,sigma,weakpix,strongpix,lowthresh,highthresh)
detector=CED.cannyedge(images,5,2,100,255,.05,.17)

finalIMAGE=detector.detect()
for a in finalIMAGE:
    plt.imshow(a,cmap = 'gray')
    plt.show()