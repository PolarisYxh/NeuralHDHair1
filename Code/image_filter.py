import cv2
from http_interface import *
try:
    from http_interface import *
except:
    from .http_interface import *
import pyfilter#have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*
import math
from skimage import transform as trans
import trimesh
try:
    from .get_face_info import get_face_info,angle2matrix
except:
    from get_face_info import get_face_info,angle2matrix
from copy import deepcopy
from get_bust import render
from scipy.spatial.transform import Rotation
def apply_matrix(v, matrix):
    '''project landmark point 3d coordinate to 2d image pixel coordinate
    '''
   
    d = np.ones(len(v))
    v = np.c_[v,d].T
    v = np.dot(matrix,v)
    for i in range(0,3):
        v[i,:] = v[i,:]/v[3,:]
    v[0,:] = (v[0,:])*0.5*(640-1)+320
    v[1,:] = -((v[1,:])*0.5)*(640-1)+320
    v[2,:] = ((v[2,:])*0.5)*(640-1)+320
    v=v[[0,1,2],:].T
    # v[np.where(np.array(landmarks)==-1)]=[0,0,0]
    return v[:,:3]

class filter_crop:
    def __init__(self,rFolder,saveFolder="",use_step=True) -> None:
        self.rFolder = rFolder
        self.conf= readjson(os.path.join(rFolder,"config.json"))
        self.saveFolder = saveFolder
        # self.face_seg = faceParsingInterface(rFolder)
        self.use_step=use_step
        if use_step:
            self.hair_step = HairStepInterface(rFolder)
        else:
            self.segall = segmentAllInterface(rFolder)
        
        self.insight_face_info = get_face_info(rFolder,False)
        #mean_landmark.json get from data/bust/color5.png,mean_landmark1.json from data/bust/body_0.png
        # self.mean_lms = np.array(readjson("mean_landmark1.json")['lms_3d'])
        self.body = trimesh.load_mesh(os.path.join(rFolder,"../female_halfbody_medium.obj"))
        self.vertices_orig = deepcopy(self.body.vertices)
        # self.lms = self.body.vertices[self.conf["body_lms"],:]
    def pyfilter2neuralhd(self,img,gender="female",image_name="",use_gt=False):
        #pyfilter输出的和hairstep： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
        #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
        self.use_gt=use_gt
        crop_image,mask = self.get_hair_seg(img,gender,image_name)
        if self.use_step or self.use_gt:
            # import torch
            # from torchvision.utils import save_image
            # crop_image1 = torch.from_numpy(crop_image.transpose(2,0,1))
            # save_image(crop_image1[[2,1,0],:,:],"test.png")
            return crop_image,mask
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
        # cv2.imshow("2",ori2D)
        # cv2.waitKey()
        return ori2D,mask
        
    def get_hair_seg(self,img,gender,image_name):
        faces, frames, framesForHair = self.insight_face_info.get_faces(img, image_name)
        pose, rot_matrix1, euler, lms_align_3d = self.insight_face_info.align_face(0)
        if self.use_step and not self.use_gt:
            # s = framesForHair[0].shape
            # framesForHair[0] = cv2.resize(framesForHair[0],(640,640))
            imgB64 = cvmat2base64(framesForHair[0])
            step = self.hair_step.request_HairStep(image_name, 'img', imgB64)

            lms_3d = self.insight_face_info.faces[0].landmark_3d_68
            
            center=np.array(self.conf["center"])
            vertices = self.vertices_orig-center
            euler[2]=0
            r = Rotation.from_euler('xyz',euler,False)
            rot_matrix = r.as_matrix()
            self.revert_rot = np.linalg.inv(rot_matrix)
            self.body.vertices = np.dot(vertices,rot_matrix)+center
            m=[]
            bust = render(self.body,bOrtho=True,matrix=m,preview_file="")
            # cv2.imshow("1",bust)
            # cv2.waitKey()
            self.mean_lms = apply_matrix(self.body.vertices[self.conf["body_lms"],:], m[0])
            
            tp = 'affine'
            tform = trans.estimate_transform(tp, lms_3d[:27,:2], self.mean_lms[:27,:2])
            step = cv2.resize(step,(640,640))
            # cv2.imshow("1",step)
            # cv2.waitKey()
            M = tform.params[0:2]
            step = cv2.warpAffine(step,
                                M, (640, 640),
                                borderValue=0.0)
            # cv2.imshow("1",step)
            # cv2.waitKey()
            step = cv2.resize(step,(256,256))
            return step,bust
        _,lms_3d = self.insight_face_info.get_lms_3d(0)
        center=np.array(self.conf["center"])
        vertices = self.vertices_orig-center
        euler[2]=0
        r = Rotation.from_euler('xyz',euler,False)
        rot_matrix = r.as_matrix()
        self.revert_rot = np.linalg.inv(rot_matrix)
        self.body.vertices = np.dot(vertices,rot_matrix)+center
        m=[]
        bust = render(self.body,bOrtho=True,matrix=m,preview_file="")
        # cv2.imshow("1",bust)
        # cv2.waitKey()
        self.mean_lms = apply_matrix(self.body.vertices[self.conf["body_lms"],:], m[0])
        tp = 'affine'
        tform = trans.estimate_transform(tp, lms_3d[:27,:2], self.mean_lms[:27,:2])
        # from util import trans_points2d
        # lms = trans_points2d(lms_3d[:,:2],tform.params)
        # drawLms(framesForHair[0],lms_3d[:,:2].astype("int"))
        M = tform.params[0:2]
        # aligned = img
        # framesForHair[0] = img
        imgB64 = cvmat2base64(framesForHair[0])
        if gender=="female":
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[56]
        else:
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[56]
        # hair_point1 = hair_point1+lms_3d[30]-lms_3d[33]
        # hair_point = [332,152]
        # lms_3d = [325,357]
        if self.use_gt:
            gt=cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/strand_map1/{image_name}")#R:（0,1）表示（向右，向左）；G：第二通道，（0,1）表示（向下，向上）
            # TODO:两种方式得到的segment图不太一样，seg中的对散发也能分割。哪个比较好 后续进行实验
            gt_parsing=gt[:,:,2].copy()
            gt_parsing[gt_parsing!=255]=0
            # gt_parsing = cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/test/seg/{image_name}")
            gt[:,:,2]=255-gt[:,:,0]
            gt[:,:,1]=255-gt[:,:,1]
            gt[:,:,0]=0
            bg = np.where((gt[:,:,2]==gt[:,:,1]) & (gt[:,:,1]==255))
            gt[bg] = [0,0,0]
            scale = 640/max(gt.shape[0],gt.shape[1])
            gt = cv2.resize(gt,(int(scale*gt.shape[1]), int(scale*gt.shape[0])))
            gt = cv2.warpAffine(gt,
                                    M, (640, 640),
                                    borderValue=0.0)
            # cv2.imshow("1",gt)
            # cv2.waitKey()
            # gt_parsing = cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/test/seg/{image_name}")
            gt_parsing = cv2.resize(gt_parsing,(int(scale*gt_parsing.shape[1]), int(scale*gt_parsing.shape[0])))
            gt_parsing = cv2.warpAffine(gt_parsing,
                                    M, (640, 640),
                                    borderValue=0.0)
            return gt,bust
        aligned = cv2.warpAffine(framesForHair[0],
                                M, (640, 640),
                                borderValue=0.0)
        masks = self.segall.request_faceParsing(image_name, 'img', imgB64,np.array([hair_point1[:2],lms_3d[30][:2]]),[1,0])#
        parsing = np.zeros_like(framesForHair[0][:,:,0])
        parsing[masks] = 255
        save_parsing = np.zeros_like(framesForHair[0])
        save_parsing[masks] = framesForHair[0][masks]
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