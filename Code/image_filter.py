import cv2
from http_interface import *
try:
    from http_interface import *
except:
    from .http_interface import *
import pyfilter#have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*
import math
from skimage import transform as trans
class filter_crop:
    def __init__(self,rFolder) -> None:
        self.rFolder = rFolder
        self.face_seg = faceParsingInterface(rFolder)
        try:
            from .get_face_info import get_face_info
        except:
            from get_face_info import get_face_info
        self.insight_face_info = get_face_info(rFolder,False)
        self.mean_lms = np.array(readjson("mean_landmark.json")['lms'])
    def pyfilter2neuralhd(self,img,image_name=""):
        #pyfilter 输出的： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
        #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
        crop_image,mask = self.get_hair_seg(img,image_name)
        mask = cv2.resize(mask,(512,512))
        crop_image= cv2.resize(crop_image,(512,512))
        ori2D=pyfilter.GetImage(crop_image)#B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示(向下）

        ori2D[:,:,2]=0
        ori2D[:,:,1]=-ori2D[:,:,1]*0.5+0.5
        ori2D[:,:,0]=1-ori2D[:,:,0]
        ori2D=ori2D[:,:,[2,1,0]]
        ori2D=(ori2D*255).astype('uint8')
        ori2D[mask==0]=[0,0,0]
        mask[mask==255]=1
        cv2.imshow("2",ori2D)
        cv2.waitKey()
        return ori2D,mask
        
    def get_hair_seg(self,img,image_name):
        faces, frames, framesForHair = self.insight_face_info.get_faces(img, image_name)
        _,lms_recon = self.insight_face_info.get_lms_forRecon(0)
        # drawLms(framesForHair[0],lms_recon.astype('int'))
        tp = 'similarity'
        tform = trans.estimate_transform(tp, lms_recon, self.mean_lms)
        M = tform.params[0:2]
        # aligned = img
        # framesForHair[0] = img
        imgB64 = cvmat2base64(framesForHair[0])
        detectedFacePart, parsing = self.face_seg.request_faceParsing(image_name, 'img', imgB64)
        aligned = cv2.warpAffine(framesForHair[0],
                                M, (640, 640),
                                borderValue=0.0)
        aligned_parsing = cv2.warpAffine(parsing,
                                M, (640, 640),
                                borderValue=0.0)
        # aligned_parsing = parsing
        aligned_parsing[(aligned_parsing!=17)]=0
        aligned_parsing[aligned_parsing==17]=255
        aligned[aligned_parsing==127]=[0,0,0]
        aligned[aligned_parsing==0]=[0,0,0]
        cv2.imshow("1",aligned)
        cv2.waitKey()
        return aligned,aligned_parsing
if __name__=="__main__":
    img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/Code/image.jpg")
    fil = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code")
    fil.pyfilter2neuralhd(img)