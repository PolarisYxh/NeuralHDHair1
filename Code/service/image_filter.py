import cv2
try:
    from http_interface import *
    from get_face_info import get_face_info,angle2matrix
    # from get_bust import render
    from inference_step import step_inference
except:
    from .http_interface import *
    from .get_face_info import get_face_info,angle2matrix
    # from .get_bust import render
    from .inference_step import step_inference

import math
from skimage import transform as trans
import trimesh

from copy import deepcopy

from dataload.render_strand import render_strand
from scipy.spatial.transform import Rotation
import torch
from torch.autograd import Variable
from Models.hourglass import Model
from Models.UNetS import Model as strandModel

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
    def __init__(self,rFolder,saveFolder="",use_step=True,use_depth=False,use_strand=False) -> None:
        # use_step (bool, optional): 网络直接生成分割图和方向图，151 docker restart segmentall-server. Defaults to True.
        # use_depth (bool, optional): 需要使用归一化后的头发深度图作为输入. Defaults to False.
        # use_strand (bool, optional): segmentanything网络直接生成分割图,本地生成方向图，151 docker restart segmentall-server. Defaults to False.
        self.rFolder = rFolder
        self.conf= readjson(os.path.join(rFolder,"service/config.json"))
        self.saveFolder = saveFolder
        # self.face_seg = faceParsingInterface(rFolder)
        self.use_step=use_step
        if use_step:#使用unet 分割及获得方向图
            self.hair_step = step_inference(os.path.join(rFolder,".."))
        else:
            self.segall = segmentAllInterface(rFolder)
        self.use_strand = use_strand#使用hairstep标准流程，用sam分割及hairstep strand生成方向图
        if use_strand:
            self.strandmodel = strandModel().cuda()
            # self.strandmodel = torch.nn.DataParallel(self.strandmodel)
            self.strandmodel.load_state_dict(torch.load(os.path.join(rFolder,"../checkpoints/img2strand.pth")))
            self.strandmodel.eval()
        self.use_depth = use_depth
        if use_depth:
            self.depthPred = Model().cuda()
            self.depthPred = torch.nn.DataParallel(self.depthPred)
            self.depthPred.load_state_dict(torch.load(os.path.join(rFolder,"../checkpoints/img2depth.pth")))
            self.depthPred.eval()
            # from img2depth import hairDepth
            # self.depthPred=hairDepth(os.path.join(rFolder,"../checkpoints/img2depth.pth"))
        self.insight_face_info = get_face_info(rFolder,False)
        #mean_landmark.json get from data/bust/color5.png,mean_landmark1.json from data/bust/body_0.png
        # self.mean_lms = np.array(readjson("mean_landmark1.json")['lms_3d'])
        self.body = trimesh.load_mesh(os.path.join(rFolder,"../female_halfbody_medium.obj"))
        self.vertices_orig = deepcopy(self.body.vertices)
        self.body_only = trimesh.load_mesh(os.path.join(rFolder,"../body.obj"))
        self.body_only_vertices_orig = deepcopy(self.body_only.vertices)
        # self.lms = self.body.vertices[self.conf["body_lms"],:]
    def get_depth(self,img,mask):
        with torch.no_grad():
            rgb_img = img[:,:, 0:3] / 255.
            rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()
            depth_pred = self.depthPred(rgb_img)
            depth_pred = depth_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            depth_pred = cv2.resize(depth_pred,(256,256))
            mask=cv2.resize(mask,(256,256))
            depth_pred_masked = depth_pred[:, :] * mask - (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred)))
            max_val = np.nanmax(depth_pred_masked)
            min_val = np.nanmin(depth_pred_masked + 2 * (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred))))
            depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val)*mask
            depth_pred_norm = np.clip(depth_pred_norm, 0., 1.)
            cv2.imwrite("norm_depth.png",(depth_pred_norm*255).astype('uint8'))
            return depth_pred_norm
    def pyfilter2neuralhd(self,img,gender="female",image_name="",use_gt=False):
        #pyfilter输出的和hairstep： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
        #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
        self.use_gt=use_gt
        crop_image,bust,img1 = self.get_hair_seg(img,gender,image_name)
        if self.use_step or self.use_gt:
            avg_color,image=self.get_hair_avgcolor1(img1,crop_image)
            
            # crop_image[np.where(parse<0.8)]=[0,0,0]
            # crop_image[:, :, 2]=0
            # crop_image = np.clip(crop_image,0,1)
            # crop_image = 1-crop_image
            # crop_image[:, :, 2]=0
            # crop_image=(crop_image*255).astype('uint8')
            cv2.imwrite("1.png",(crop_image*255).astype('uint8'))
            avg_color=np.append(avg_color,255)
            return crop_image,bust,avg_color,image,self.revert_rot
        if self.use_strand:
            crop_image= cv2.resize(crop_image,(512,512))
            mask1 = cv2.cvtColor(crop_image,cv2.COLOR_RGB2GRAY)
            mask2 = np.copy(mask1)
            mask2[mask2>0]=1
            crop_image2 = np.copy(crop_image)
            crop_image2[mask2==0]=[0,0,0]
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
            mask2 = cv2.erode(mask2,kernel)
            crop_image1 = np.copy(crop_image)
            crop_image1[mask2==0]=[0,0,0]
            
            avg_color = (np.sum(np.sum(crop_image1,axis=0),axis=0)/np.sum(mask2>0)).astype('int')
            crop_image1[mask2==0]=avg_color
            # cv2.imshow("4",crop_image1)
            # cv2.waitKey()
            mask1[mask1>0] = 255 
            crop_image = crop_image/255
            crop_image = Variable(torch.from_numpy(crop_image).permute(2, 0, 1).float().unsqueeze(0)).cuda()
            strand_pred = self.strandmodel(crop_image)
            strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)  # 512 * 512 *60
            x = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
            x[:,:,1:3]=strand_pred
            x[:,:,1]=1-x[:,:,1]
            x[:,:,2]=1-x[:,:,2]
            x[mask1==0]=[0,0,0]
            x=(x*255).astype('uint8')
            avg_color=np.append(avg_color,255)
            return x,bust,avg_color,crop_image2,self.revert_rot
            # strand_pred = np.concatenate([mask+body*0.5, strand_pred*mask], axis=-1)
        import pyfilter#have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*
        avg_color,image1=self.get_hair_avgcolor(img1,crop_image)
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
        avg_color=np.append(avg_color,255)
        # cv2.imshow("2",ori2D)
        # cv2.waitKey()
        return ori2D,mask,avg_color,image1,self.revert_rot
    def get_hair_avgcolor1(self,img,mask):
        parse = mask[:, :, 2]
        mask1=mask[:, :, [0, 1]]
        mask1[np.where(parse<0.8)]=[0,0]
        
        mask1=cv2.resize(mask1,(img.shape[1],img.shape[0]))
        mask1=np.sum(mask1,axis=2)
        mask1[mask1>0]=1
        image = np.copy(img)
        image[mask1==0]=[0,0,0]
    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        mask1 = cv2.erode(mask1,kernel)
        crop_image1=cv2.resize(np.copy(img),mask1.shape[:2])
        crop_image1[mask1==0]=[0,0,0]
        avg_color = np.sum(np.sum(crop_image1,axis=0),axis=0)/np.sum(mask1>0)
        crop_image1[mask1==0]=avg_color
        return avg_color.astype('int'),image
    def get_hair_avgcolor(self,img,mask):
        mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        mask[mask>0]=1
        image = np.copy(img)
        image[mask==0]=[0,0,0]
    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        mask = cv2.erode(mask,kernel)
        crop_image1=cv2.resize(np.copy(img),mask.shape[:2])
        crop_image1[mask==0]=[0,0,0]
        avg_color = np.sum(np.sum(crop_image1,axis=0),axis=0)/np.sum(mask>0)
        crop_image1[mask==0]=avg_color
        # cv2.imshow("4",image)
        # cv2.waitKey()
        return avg_color.astype('int'),image
    def get_hair_seg(self,img,gender,image_name):
        faces, frames, framesForHair = self.insight_face_info.get_faces(img, image_name)
        pose, rot_matrix1, euler, lms_align_3d = self.insight_face_info.align_face(0)
        lms_3d = self.insight_face_info.faces[0].landmark_3d_68
        
        center=np.array(self.conf["center"])
        vertices = self.vertices_orig-center
        body_only_vertices = self.body_only_vertices_orig-center
        euler[2]=0
        r = Rotation.from_euler('xyz',euler,False)
        rot_matrix = r.as_matrix()
        self.revert_rot = np.linalg.inv(rot_matrix)
        self.body.vertices = np.dot(vertices,rot_matrix)+center
        self.body_only.vertices = np.dot(body_only_vertices,rot_matrix)+center
        m=[]
        _,bust,img2 = render_strand([[]],[],self.body,orientation=[],intensity=3,matrix=m,mask=True)
        # cv2.imshow("1",bust)
        # cv2.waitKey() 
        self.mean_lms = apply_matrix(self.body.vertices[self.conf["body_lms"],:], m[0])
        shoulder_lms = np.array(self.conf["shoulder_lms"])
        self.shoulder_lms = apply_matrix(self.body_only.vertices[shoulder_lms,:], m[0])
        tp = 'affine'
        tform = trans.estimate_transform(tp, lms_3d[:27,:2], self.mean_lms[:27,:2])
        M = tform.params[0:2]
        if self.use_step and not self.use_gt:
            # s = framesForHair[0].shape
            # framesForHair[0] = cv2.resize(framesForHair[0],(640,640))
            step = self.hair_step.inference(framesForHair[0])
            step = cv2.resize(step,(640,640))
            # cv2.imshow("1",step)
            # cv2.waitKey()
            step = cv2.warpAffine(step,
                                M, (640, 640),
                                borderValue=0.0)
            img1 = cv2.warpAffine(framesForHair[0],
                                M, (640, 640),
                                borderValue=0.0)
            # cv2.imshow("1",step)
            # cv2.waitKey()
            
            # get warp
            # step1 = np.copy(step)
            # parse = step1[:, :, 2]
            # mask1=step1[:, :, [0, 1]]
            # mask1[np.where(parse<0.8)]=[0,0]
            
            # mask1=np.sum(mask1,axis=2)
            # parse[mask1>0]=0
            # x=np.clip(parse,0,1)
            # parse = np.clip(parse,0,1)*255
            # # 这里parse是脸和身体的分割图
            # _, thresh = cv2.threshold(parse, 127, 255, cv2.THRESH_BINARY)
            # # thresh=thresh.astype('uint8')
            # # Define a structuring element
            # kernel = np.ones((5,5), np.uint8)
            # # Apply erosion on the image
            # thresh = cv2.erode(thresh, kernel, iterations=1)
            # thresh = cv2.dilate(thresh, kernel, iterations=1)
            # thresh=thresh.astype('uint8')
            # cv2.imwrite('parse.png', thresh.astype('uint8'))
            # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # m_l=0
            # index = -1
            # for i,contour in enumerate(contours):
            #     l = len(contour)
            #     if l>m_l:
            #         m_l=l
            #         index=i
            # contour = contours[index]
            # # Calculate epsilon based on the contour perimeter
            # epsilon = 0.0005 * cv2.arcLength(contour, True)
            # # Approximate the contour
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            # # Draw the approximated contour on the original image
            # crop_image1=cv2.drawContours((step*255).astype('uint8'), [approx], -1, (0, 255, 0), 1)
            # # Display the image
            # cv2.imwrite('Image.png', crop_image1)
            
            
            # from scipy.spatial import KDTree
            # kdtree = KDTree(approx[:,0,:])
            # dist, index1 = kdtree.query(self.shoulder_lms[:,:2])
            # real_lms = np.append(approx[:,0,:][index1],self.mean_lms[:,:2],axis=0)
            # drawLms((step*255).astype('uint8'),real_lms.astype('int'))
            # target_lms = np.append(self.shoulder_lms[:,:2],self.mean_lms[:,:2],axis=0)
            # drawLms((step*255).astype('uint8'),target_lms.astype('int'))
            # tform = trans.estimate_transform(tp, real_lms, target_lms)
            # M1 = tform.params[0:2]
            # step = cv2.warpAffine(step,
            #                     M1, (640, 640),
            #                     borderValue=0.0)

            step = cv2.resize(step,(256,256))
            return step,bust,img1
        if self.use_gt:
            gt=cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/strand_map/{image_name}")#R:（0,1）表示（向右，向左）；G：第二通道，（0,1）表示（向下，向上）
            # TODO:两种方式得到的segment图不太一样，seg中的对散发也能分割。哪个比较好 后续进行实验
            # gt_parsing=gt[:,:,2].copy()
            # gt_parsing[gt_parsing!=255]=0
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
            gt = cv2.resize(gt,(256,256))
            return gt,bust,framesForHair[0]
        if gender=="female":
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[33]
        else:
            hair_point1 = (lms_3d[21]+lms_3d[22])-lms_3d[33]
        # hair_point1 = hair_point1+lms_3d[30]-lms_3d[33]
        # hair_point = [332,152]
        # lms_3d = [325,357]
        imgB64 = cvmat2base64(framesForHair[0])
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
        # cv2.imshow("1",aligned)
        # cv2.waitKey()
        return aligned,bust,framesForHair[0]
if __name__=="__main__":
    gender = ['male','female']
    test_dir="/home/yangxinhang/NeuralHDHair/data/test/paper"
    file_names = os.listdir(test_dir)
    for name in file_names[1:]:
        test_file = os.path.join(test_dir,name)
        img = cv2.imread(test_file)
        # img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/Bust/body_0.png")
        fil = filter_crop(os.path.join(os.path.dirname(__file__),"../"),\
                            os.path.join(os.path.dirname(__file__),"../data/test"),\
                            use_step=True,use_depth=True,use_strand=False)
        fil.pyfilter2neuralhd(img,image_name=name)