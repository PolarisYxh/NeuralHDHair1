import cv2
import os
dir = "data/test"
imgs = os.listdir(dir)
for x in imgs:
    if "Screenshot" in x and "_1" not in x:
        img = cv2.imread(os.path.join(dir,x))
        img = img[80:1000,750:1500,:]
        cv2.imshow("1",img)
        cv2.waitKey()
        cv2.imwrite(os.path.join(dir,x[:-4]+"_1.png"),img)