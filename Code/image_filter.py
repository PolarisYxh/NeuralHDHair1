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
    def __init__(self,rFolder,saveFolder="") -> None:
        self.rFolder = rFolder
        self.saveFolder = saveFolder
        # self.face_seg = faceParsingInterface(rFolder)
        self.segall = segmentAllInterface(rFolder)
        try:
            from .get_face_info import get_face_info
        except:
            from get_face_info import get_face_info
        self.insight_face_info = get_face_info(rFolder,False)
        #mean_landmark.json get from data/bust/color5.png,mean_landmark1.json from data/bust/body_0.png
        self.mean_lms = np.array(readjson("mean_landmark1.json")['lms_3d'])
    def pyfilter2neuralhd(self,img,gender="female",image_name=""):
        #pyfilter 输出的： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
        #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
        crop_image,mask = self.get_hair_seg(img,gender,image_name)
        mask = cv2.resize(mask,(512,512))
        crop_image= cv2.resize(crop_image,(512,512))
        ori2D=pyfilter.GetImage(crop_image)#B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示(向下）

        ori2D[:,:,2]=0
        ori2D[:,:,1]=-ori2D[:,:,1]*0.5+0.5
        ori2D[:,:,0]=1-ori2D[:,:,0]
        ori2D=ori2D[:,:,[2,1,0]]
        ori2D=(ori2D*255).astype('uint8')
        ori2D[mask==0]=[0,0,0]
        # bust = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/Bust/body_0.png")
        # bust = cv2.resize(bust,(512,512))
        ori2D[mask==0]=0
        mask[mask==255]=1
        # ori2D[mask==0]=bust[mask==0]
        cv2.imwrite(os.path.join(self.saveFolder,gender+"_out",image_name[:-4]+"_mask.png"),ori2D)
        cv2.imshow("2",ori2D)
        cv2.waitKey()
        return ori2D,mask
        
    def get_hair_seg(self,img,gender,image_name):
        faces, frames, framesForHair = self.insight_face_info.get_faces(img, image_name)
        _,lms_3d = self.insight_face_info.get_lms_3d(0)
        tp = 'affine'
        bust = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/Bust/body_0.png")
        bust = cv2.resize(bust,(640,640))
        # drawLms(bust,self.mean_lms[:,:2].astype("int"))
        tform = trans.estimate_transform(tp, lms_3d[:27,:2], self.mean_lms[:27,:2])
        # from util import trans_points2d
        # lms = trans_points2d(lms_3d[:,:2],tform.params)
        # drawLms(framesForHair[0],lms_3d[:,:2].astype("int"))
        M = tform.params[0:2]
        # aligned = img
        # framesForHair[0] = img
        imgB64 = cvmat2base64(framesForHair[0])
        # detectedFacePart, parsing = self.face_seg.request_faceParsing(image_name, 'img', imgB64)
        if gender=="female":
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[56]
        else:
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[56]
        # hair_point1 = hair_point1+lms_3d[30]-lms_3d[33]
        # hair_point = [332,152]
        # lms_3d = [325,357]
        masks = self.segall.request_faceParsing(image_name, 'img', imgB64,np.array([hair_point1[:2],lms_3d[30][:2]]),[1,0])#
        parsing = np.zeros_like(framesForHair[0][:,:,0])
        parsing[masks] = 255
        save_parsing = np.zeros_like(framesForHair[0])
        save_parsing[masks] = framesForHair[0][masks]
        aligned = cv2.warpAffine(framesForHair[0],
                                M, (640, 640),
                                borderValue=0.0)
        aligned_parsing = cv2.warpAffine(parsing,
                                M, (640, 640),
                                borderValue=0.0)
        aligned_parsing[(aligned_parsing!=0)]=255
        aligned[aligned_parsing==0]=[0,0,0]
        # cv2.imwrite(os.path.join(self.saveFolder,gender+"_out",image_name[:-4]+"_parse.png"),save_parsing)
        # cv2.imshow("1",save_parsing)
        # cv2.waitKey()
        return aligned,aligned_parsing
if __name__=="__main__":
    gender = ['male','female']
    test_dir="/home/yxh/Documents/company/NeuralHDHair/data/test"
    for g in gender:
        test_dir1 = os.path.join(test_dir,g)
        file_names = os.listdir(test_dir1)
        for name in file_names[:]:
            # name = "9.jpg"
            test_file = os.path.join(test_dir1,name)
            img = cv2.imread(test_file)
            # img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/Bust/body_0.png")
            fil = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code",test_dir)
            fil.pyfilter2neuralhd(img,g,name)