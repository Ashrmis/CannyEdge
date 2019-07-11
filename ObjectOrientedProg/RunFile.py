import CannyEdgeDetector_OOP as CED
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
images=[]
path= r'C:PATH_HERE'
imagetypes=['.jpg','.png','.jpeg']

for f in os.listdir(path):
    extension=os.path.splitext(f)[1]
    if extension not in imagetypes:
        continue
    images.append(Image.open(os.path.join(path,f)))

# image,size,sigma,weakpix,strongpix,lowthresh,highthresh)
detector=CED.cannyedge(images,5,2,100,255,.05,.17)

finalIMAGE=detector.detect()
for a in finalIMAGE:
    plt.imshow(a,cmap = 'gray')
    plt.show()