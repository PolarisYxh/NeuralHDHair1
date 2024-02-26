import cv2
try:
    from http_interface import *
    from get_face_info import get_face_info,angle2matrix
    from inference_step import step_inference
except:
    from .http_interface import *
    from .get_face_info import get_face_info,angle2matrix
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
        # use_step (bool, optional): 使用自己训练的网络得到初步的身体分割图和头发方向图，否则使用faceparsing分割得到身体和头发分割图，151 docker restart segmentall-server. Defaults to True.
        # use_depth (bool, optional): 需要使用归一化后的头发深度图作为输入. Defaults to False.
        # use_strand (bool, optional): segmentanything网络直接生成分割图,本地生成方向图，151 docker restart segmentall-server. Defaults to False.
        self.rFolder = rFolder
        self.conf= readjson(os.path.join(rFolder,"service/config.json"))
        self.saveFolder = saveFolder

        self.segall = segmentAllInterface(rFolder)
        self.use_step=use_step
        if use_step:#使用unet 分割及获得方向图
            self.hair_step = step_inference(os.path.join(rFolder,".."))
        else:
            self.face_seg = faceParsingInterface(rFolder)
        self.use_strand = use_strand#使用hairstep标准流程，用sam分割及hairstep strand生成方向图
        if use_strand:
            self.strandmodel = strandModel().cuda()
            # self.strandmodel = torch.nn.DataParallel(self.strandmodel) #第一次新增数据后267000；第二次新增：205002,1150002；只用短发训练：30000
            self.strandmodel.load_state_dict(torch.load(os.path.join(rFolder,"../checkpoints/img2strand-1150002.pth")))
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
        self.body = trimesh.load_mesh(os.path.join(rFolder,"../female_halfbody_medium_join.obj"))
        self.vertices_orig = deepcopy(self.body.vertices)
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
            # cv2.imwrite("norm_depth.png",(depth_pred_norm*255).astype('uint8'))
            return depth_pred_norm
    def pyfilter2neuralhd(self,img,gender="female",image_name="",use_gt=False):
        """_summary_

        Args:
            img (_type_): _description_
            gender (str, optional): _description_. Defaults to "female".
            image_name (str, optional): _description_. Defaults to "".
            use_gt (bool, optional): 分割图用gt图. Defaults to False.

        Returns:
            strand2d: 方向图，R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
            bust:人体rgb图
            avg_color:头发平均颜色
            crop_image2:对齐到标准头后的头发分割图
            revert_rot,:人脸欧拉角
            self.cam_intri:相机内参矩阵
            self.cam_extri:相机外参矩阵
        """        
        #pyfilter输出的和hairstep数据集中方向图说明： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（向上，向下）;R:255
        #neuralhd即该函数输出的方向图说明：          R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）;B:0
        self.use_gt=use_gt
        crop_image,bust,img1 = self.get_hair_seg(img,gender,image_name)
        if self.use_gt:
            avg_color,image=self.get_hair_avgcolor1(img1,crop_image)
            
            # crop_image[np.where(parse<0.8)]=[0,0,0]
            # crop_image[:, :, 2]=0
            # crop_image = np.clip(crop_image,0,1)
            # crop_image = 1-crop_image
            # crop_image[:, :, 2]=0
            # crop_image=(crop_image*255).astype('uint8')
            # cv2.imwrite("1.png",(crop_image*255).astype('uint8'))
            avg_color=np.append(avg_color,255)
            return crop_image,bust,avg_color,image,self.revert_rot,self.cam_intri,self.cam_extri
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
            # hairstep 数据集中的方向图，用于对比，debug
            # strand2d = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
            # strand2d[:,:,:2]=strand_pred[:,:,[1,0]]
            # strand2d[:,:,2]=1.0
            # strand2d[mask1==0]=[0,0,0]
            # strand2d=(strand2d*255).astype('uint8')
            # cv2.imwrite(image_name.split('.')[0]+"_ori2_0.png",strand2d)
            # 方向图网络需要输入的方向图
            strand2d = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
            strand2d[:,:,1:3]=strand_pred#strand_pred:0通道
            strand2d[:,:,1]=1-strand2d[:,:,1]
            strand2d[:,:,2]=1-strand2d[:,:,2]
            strand2d[mask1==0]=[0,0,0]
            strand2d=(strand2d*255).astype('uint8')
            # cv2.imwrite(image_name.split('.')[0]+"_ori2.png",strand2d)
            avg_color=np.append(avg_color,255)
            return strand2d,bust,avg_color,crop_image2,self.revert_rot,self.cam_intri,self.cam_extri
            # strand_pred = np.concatenate([mask+body*0.5, strand_pred*mask], axis=-1)
        import pyfilter#first generate it in strandhair HairNet_orient2D repo;have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*，已经被弃用
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
        return ori2D,mask,avg_color,image1,self.revert_rot,self.cam_intri,self.cam_extri
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
        euler[2]=0
        r = Rotation.from_euler('xyz',euler,False)
        rot_matrix = r.as_matrix()
        self.revert_rot = np.linalg.inv(rot_matrix)
        self.body.vertices = np.dot(vertices,rot_matrix)+center
        m=[]
        _,bust,img2 = render_strand([[]],[],self.body,inference=True,orientation=[],intensity=3,matrix=m,mask=True)
        # calculate cam intrinsic and cam extrinsic
        self.cam_intri = m[1]#使用奇胜人脸内参的相机，从相机空间到裁剪空间
        x1=trans.SimilarityTransform(translation=-center,dimensionality=3).params
        x2=trans.SimilarityTransform(translation=center,dimensionality=3).params
        four_by_four = np.eye(4)  
        four_by_four[:3, :3] = rot_matrix 
        four_by_four[:-1, -1] = 0  
        four_by_four[-1, :-1] = 0  
        four_by_four[-1, -1] = 1  
        self.cam_extri = m[2]@x2@np.linalg.inv(four_by_four)@x1 #使用奇胜人脸内参的相机，从世界坐标到相机空间的变换矩阵
        self.cam_pose = x2@four_by_four@x1@np.linalg.inv(m[2])#使用奇胜人脸内参的相机，相机的世界坐标系位置,m[2] 存疑
        # cv2.imshow("1",bust)
        # cv2.waitKey() 
        self.mean_lms = apply_matrix(self.body.vertices[self.conf["body_lms"],:], m[0])#得到自己正交相机标准脸关键点的图像投影
        shoulder_lms = np.array(self.conf["shoulder_lms"])
        self.shoulder_lms = apply_matrix(self.body.vertices[shoulder_lms,:], m[0])
        tp = 'affine'
        tform = trans.estimate_transform(tp, lms_3d[:17,:2], self.mean_lms[:17,:2])
        M = tform.params[0:2]
        if self.use_step:#先粗糙分割头发
            framesForHair[0] = cv2.resize(framesForHair[0],(640,640))
            step = self.hair_step.inference(framesForHair[0])
            step = cv2.resize(step,(640,640))
            step_align = cv2.warpAffine(step,
                                M, (640, 640),
                                borderValue=0.0)
            # step_align1=(step_align*255).astype('uint8')
            # lms1 = trans.matrix_transform(lms_3d[:17,:2], tform.params)
            # for point in lms1:
            #     cv2.circle(step_align1, tuple(point.astype('int')), 1, (0, 255, 0), 2)
            # for point in self.mean_lms[:17,:2]:
            #     cv2.circle(step_align1, tuple(point.astype('int')), 1, (255, 0, 0), 2)
            # cv2.imwrite(image_name.split('.')[0]+"_2.png",step_align1)
            
            #身体分割图
            body_parse = step_align[:, :, 2]
            mask1=step_align[:, :, [0, 1]]
            mask1[np.where(body_parse<0.8)]=[0,0]
            mask1=np.sum(mask1,axis=2)
            body_parse[mask1>0]=0
            body_parse = np.clip(body_parse,0,1)*255
            
            #mask1:头发分割图，使用sam分割，需要先找到头发的点位
            parse = step[:, :, 2]
            mask1=step[:, :, [0, 1]]
            mask1[np.where(parse<0.8)]=[0,0]
            mask1=np.sum(mask1,axis=2)
            mask1[mask1>0]=255
            # mask1[(lms_3d[21][1]+lms_3d[22][1])//2:,:] = 0
        else:
            framesForHair[0] = cv2.resize(framesForHair[0],(640,640))
            imgB64 = cvmat2base64(framesForHair[0])
            detectedFacePart, parsing = self.face_seg.request_faceParsing(image_name, 'img', imgB64)
            #身体分割图
            body_parse = np.zeros_like(parsing.astype('uint8'))
            for part in ["cloth","neck","necklace"]:
                if part in detectedFacePart:
                    body_parse[(parsing.astype('uint8')[:, :] == detectedFacePart[part])] = 250
            kernel = np.ones((5,5),np.uint8)
            body_parse = cv2.erode(body_parse.astype('uint8'),kernel,iterations = 1)
            body_parse = cv2.dilate(body_parse.astype('uint8'),kernel,iterations = 1)
            body_parse = cv2.warpAffine(body_parse,
                                M, (640, 640),
                                borderValue=0.0)
            #mask1:头发分割图
            mask1 = parsing.astype('uint8').copy()
            mask1[mask1[:, :] != detectedFacePart["hair"]] = 0  #,1,0
            mask1[mask1[:, :] == detectedFacePart["hair"]] = 255
        cv2.imwrite(image_name.split('.')[0]+"_mask.png",mask1)
        # cv2.imwrite(image_name.split('.')[0]+"_mask1.png",body_parse)
        kernel = np.ones((17,17),np.uint8)
        mask1 = cv2.erode(mask1.astype('uint8'),kernel,iterations = 1)
        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = []
        for contour in contours:  
            area = cv2.contourArea(contour)  
            areas.append(area)
        m_area = np.mean(np.array(areas))*0.618
        index = np.where(areas>m_area)
        x=index[0].tolist()
        contours1 = np.array(contours,dtype=object)[x]
        hair_point1 = []
        for hair_area1 in contours1:
            # cv2.imwrite(image_name.split('.')[0]+"_mask.png",mask1)
            # 找中位的点,先找到横向的中间，再找到纵向的中间
            # hair_area1 = np.array(hair_area1[:,0,:])
            # sorted_indices = np.argsort(hair_area1[:, 1])
            # sorted_points = hair_area1[sorted_indices]
            # m_y = sorted_points[len(sorted_points)//2][1]
            # y=sorted_points[sorted_points[:,1]==m_y]
            # hair_area1 = np.where(mask1>0)
            # sorted_indices = np.argsort(y[:, 0])
            # sorted_points = y[sorted_indices]
            # subarrays = np.split(sorted_points[:,0], np.where(np.diff(sorted_points[:,0]) != 1)[0] + 1)#分段连续数组
            # max_subarray = max(subarrays, key=len)
            # m_x=max_subarray[len(max_subarray)//2]
            # hair_point1.append([m_y,m_x])
            hair_area1 = np.array(hair_area1[:,0,:])
            sorted_indices = np.argsort(hair_area1[:, 0])
            sorted_points = hair_area1[sorted_indices]
            m_y = sorted_points[len(sorted_points)//2][0]
            y=sorted_points[sorted_points[:,0]==m_y]
            p = np.sort(y[:, 1])
            subarrays = np.split(p, np.where(np.diff(p) != 1)[0] + 1)#分段连续数组
            if len(subarrays)>1:
                m_x = int((subarrays[0][0]+subarrays[1][0])//2)
            else:
                m_x = int((np.max(y[:, 1])+np.min(y[:, 1]))//2)
            hair_point1.append([m_y,m_x])
            
        imgB64 = cvmat2base64(framesForHair[0])#framesForHair[0] 640,640
        aligned = cv2.warpAffine(framesForHair[0],
                                M, (640, 640),
                                borderValue=0.0)#头发转到标准位置
        # cv2.imwrite(image_name.split('.')[0]+"_3.png",aligned)
        hair_point1 = np.array(hair_point1)
        hair_point2 = np.append(hair_point1,lms_3d[[30,33],:2],axis=0)
        labels = np.zeros(len(hair_point1)+2).astype('int')
        labels[:len(hair_point1)]=1
        masks = self.segall.request_faceParsing(image_name, 'img', imgB64,hair_point2,labels)
        labels = np.zeros(len(hair_point1)+1).astype('int')
        labels[:len(hair_point1)]=1
        if masks[tuple(lms_3d[27,[1,0]].astype('int').T)]==True:
            hair_point2 = np.append(hair_point1,lms_3d[30,:2][None],axis=0)
            masks = self.segall.request_faceParsing(image_name, 'img', imgB64,hair_point2,labels)
        if masks[tuple(lms_3d[27,[1,0]].astype('int').T)]==True:
            hair_point2 = np.append(hair_point1,lms_3d[27,:2][None],axis=0)
            masks = self.segall.request_faceParsing(image_name, 'img', imgB64,hair_point2,labels)
        if masks[tuple(lms_3d[27,[1,0]].astype('int').T)]==True:
            hair_point2 = np.append(hair_point1,lms_3d[33,:2][None],axis=0)
            masks = self.segall.request_faceParsing(image_name, 'img', imgB64,hair_point2,labels)
        parsing = np.zeros_like(framesForHair[0][:,:,0])
        parsing[masks] = 255
        save_parsing = np.zeros_like(framesForHair[0])
        save_parsing[masks] = framesForHair[0][masks]
        x=np.copy(framesForHair[0])
        for contour in contours1:
            for point in contour[:,0,:]:
                cv2.circle(x, tuple(point), 1, (0, 255, 0), 2)
        for point in hair_point1:
            cv2.circle(x, tuple(point), 1, (255, 0, 0), 2)
        # cv2.imwrite(image_name.split('.')[0]+"_2.png",parsing)
        aligned_parsing = cv2.warpAffine(parsing,
                                M, (640, 640),
                                borderValue=0.0)
        aligned_parsing[(aligned_parsing!=0)]=255
        aligned[aligned_parsing==0]=[0,0,0]
        aligned_mask=np.sum(aligned,axis=2)
        hair_area=np.where(aligned_mask!=0)
        M1 = np.identity(2)
        M1 = np.append(M1,np.array([[0,0]]).T,axis=1)
        # cv2.imwrite("lms2.png",aligned)
        # cv2.imwrite("lms3.png",(step_align*255).astype('uint8'))
        # get warp to align standard shoulder and people's shoulder
        
        if np.max(hair_area[0])>self.mean_lms[8,1] and np.max(aligned_mask[tuple(self.shoulder_lms[:,[1,0]].T.astype('int'))])==0:#头发长于下巴且短于肩膀，需要在尺度上对齐肩膀和头发
            # 这里parse是脸和身体的分割图
            _, thresh = cv2.threshold(body_parse, 127, 255, cv2.THRESH_BINARY)
            # Define a structuring element
            kernel = np.ones((5,5), np.uint8)
            # Apply erosion on the image
            thresh = cv2.erode(thresh, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh=thresh.astype('uint8')
            # cv2.imwrite('parse.png', thresh.astype('uint8'))
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour = max(contours,key=cv2.contourArea)
            # Calculate epsilon based on the contour perimeter
            epsilon = 0.0001 * cv2.arcLength(contour, True)
            # Approximate the contour
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Draw the approximated contour on the original image
            # crop_image1=cv2.drawContours(body_parse.astype('uint8'), [approx], -1, 150, 1)
            # Display the image
            # cv2.imwrite('lms.png', crop_image1)
            
            #在轮廓中找到最接近的关键点
            from scipy.spatial import KDTree
            kdtree = KDTree(approx[:,0,:])
            dist, index1 = kdtree.query(self.shoulder_lms[[3,4],:2])
            idx = [0,16] #0,16,19,
            # real_lms = np.append(approx[:,0,:][index1],self.mean_lms[idx,:2],axis=0)
            # # drawLms((aligned*255).astype('uint8'),real_lms.astype('int'))
            # target_lms = np.append(self.shoulder_lms[[3,4],:2],self.mean_lms[idx,:2],axis=0)
            # # drawLms((aligned*255).astype('uint8'),target_lms.astype('int'),name="lms1.png")
            # tform = trans.estimate_transform(tp, real_lms, target_lms)
            # M1 = tform.params[0:2]
            # M1[0,0]=1#左右方向尺度不变，仅拉长头发到肩膀
            # M1[0,1]=0
            # M1[1,0]=0
            # M1[0,2]=0
            
            point=trans.matrix_transform(hair_point1, tform.params)
            length = np.linalg.norm(self.mean_lms[idx[0],:2]-self.mean_lms[idx[1],:2])//2
            point=np.mean(self.mean_lms[idx,:2],axis=0)[None]
            # point[0][1]-=length
            target_p = np.mean(self.shoulder_lms[[3,4],:2],axis=0)
            source_p = np.mean(approx[:,0,:][index1[[0,1]]],axis=0)
            target_p[0]=point[0][0]
            source_p[0] = point[0][0]
            real_lms = np.append(target_p[None],point,axis=0)
            # target_lms = np.append(p[None],point,axis=0)
            # drawLms((aligned*255).astype('uint8'),real_lms.astype('int'))
            # drawLms((aligned*255).astype('uint8'),target_lms.astype('int'),name="lms1.png")
            M1 = trans.AffineTransform(translation=[point[0][0],point[0][1]]).params@trans.AffineTransform(scale=[1,(target_p[1]-point[0][1])/(source_p[1]-point[0][1])]).params@trans.AffineTransform(translation=[-point[0][0],-point[0][1]]).params
            p2 = trans.matrix_transform(source_p,M1)
            target_lms = np.append(p2,point,axis=0)
            # drawLms((aligned*255).astype('uint8'),target_lms.astype('int'),name="lms1.png")
            
            M1 = M1[0:2]
            M1[0,0]=1#左右方向尺度不变，仅拉长头发到肩膀
            M1[0,1]=0
            M1[1,0]=0
            # M1[1,1]=np.clip(M1[1,1],0.9,1.1)
            M1[0,2]=0
            # M1[1,2]=np.clip(M1[1,2],-20,20)
            logging.info('aligned to shoulder')

        aligned = cv2.warpAffine(aligned,
                                M1, (640, 640),
                                borderValue=0.0)#将对齐到标准头的头发分割图对齐肩膀
        # step_align1=(aligned*255).astype('uint8')
        # lms1 = trans.matrix_transform(lms_3d[:17,:2], tform.params)
        # for point in lms1:
        #     cv2.circle(aligned, tuple(point.astype('int')), 1, (0, 255, 0), 2)
        # for point in self.mean_lms[:17,:2]:
        #     cv2.circle(aligned, tuple(point.astype('int')), 1, (255, 0, 0), 2)
        # cv2.imwrite(image_name.split('.')[0]+"_2.png",aligned)
        # for debug
        bust1 = cv2.resize(bust,(640,640))
        aligned1 = np.copy(aligned)
        aligned_parsing1 = cv2.warpAffine(aligned_parsing,
                                M1, (640, 640),
                                borderValue=0.0)
        cv2.imwrite("1.png",aligned_parsing1)
        aligned1[(bust1>0)&(aligned_parsing1==0)]=[0,0,255]
        cv2.imwrite(image_name.split('.')[0]+"_parse.png",aligned1)
        return aligned,bust,framesForHair[0]
if __name__=="__main__":
    gender = ['male','female']
    test_dir="./data/test/strand2d"
    file_names = os.listdir(test_dir)
    test_dir="./data/test"
    file_names = ["image1.png"]
    os.environ['CUDA_VISIBLE_DEVICES'] ="4"
    
    # body = trimesh.load_mesh("/app/female_halfbody_medium_join.obj")
    # body.visual = body.visual.to_color()
    # body.visual.vertex_colors = np.array([0, 0, 0, 255])
    # _,bust,img2 = render_strand([[]],[],body,orientation=[],intensity=3,mask=True)
    # cv2.imwrite("1.png",(bust*255).astype('uint8'))
    for name in file_names:
        # name = "female_12.jpg"
        test_file = os.path.join(test_dir,name)
        img = cv2.imread(test_file)
        # img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/Bust/body_0.png")
        fil = filter_crop(os.path.join(os.path.dirname(__file__),"../"),\
                            os.path.join(os.path.dirname(__file__),"../data/test"),\
                            use_step=True,use_depth=False,use_strand=True)
        strand,_,_,x,_,_,_ = fil.pyfilter2neuralhd(img,image_name=name)
        cv2.imwrite(name.split('.')[0]+"_ori1.png",255-strand)
        cv2.imwrite(name.split('.')[0]+"_seg1.png",x)
