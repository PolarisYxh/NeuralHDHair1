# 打开realistic-exe-linux项目可执行文件进行渲染
from options.inference_options import InferenceOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
from Tools.to_unity import *
from Code.service.image_filter import filter_crop
import cv2

import time
from tqdm import tqdm
from Tools.file_io import *
from skimage import transform as trans
from dataload.render_strand import render_strand,render_cartoon
import trimesh
from Tools.utils import timeCost
from util import cvmat2base64
import logging
from skimage import measure
from Tools.utils import transform_Inv
class strand_inference:
    def __init__(self,rFolder,HairFilterLocal=False,use_modeling=False,use_ori_addinfo=False,use_hd=False,use_step=True,use_strand=False,Bidirectional_growth=False,gpu_ids=[],get_cartoon=False) -> None:
        """_summary_

        Args:
            rFolder (_type_): _description_
            use_step (bool, optional): 网络直接生成分割图和方向图，151 docker restart segmentall-server. Defaults to True.
            use_ori_addinfo (bool, optional): 使用方向图作为ModelingHD网络的输入，否则使用归一化后的头发深度图作为输入. Defaults to False.
            use_strand (bool, optional): segmentanything网络直接生成分割图,本地生成方向图，151 docker restart segmentall-server. Defaults to False.
            Bidirectional_growth (bool, optional): _description_. Defaults to False.
            use_modeling:表示要使用到 输入深度信息的Modeling网络
        """   
        self.get_cartoon = get_cartoon
        self.use_strand = use_strand
        if use_modeling and not use_ori_addinfo:
            use_depth=True   
        else:
            use_depth=False
        self.use_depth = use_depth
        self.HairFilterLocal = HairFilterLocal 
        if  self.HairFilterLocal:
            self.img_filter = filter_crop(os.path.dirname(__file__),\
                                        os.path.join(os.path.dirname(__file__),"../data/test"),\
                                        use_step=use_step,use_depth=use_depth,use_strand=use_strand)
        else:
            from http_interface import HairFilterInterface
            self.img_filter = HairFilterInterface(os.path.dirname(__file__))
        self.use_step=use_step
        self.opt=InferenceOptions().initialize(use_modeling=use_modeling)
        self.iter = {}
        self.delete_far = False
        if self.delete_far:
            from Tools.connect_root import find_root
            self.froot = find_root(rFolder)
        self.opt.gpu_ids=gpu_ids
        gpu_str = [str(i) for i in self.opt.gpu_ids]
        gpu_str = ','.join(gpu_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        self.opt.current_path = rFolder
        self.opt.name="2023-05-06_bust_test"
        self.opt.save_dir="data/Train_input"
        self.opt.is_Train = False
        self.use_modeling = use_modeling
        self.use_hd = use_hd
        if use_modeling:
            self.opt.voxel_size = "192,256,256"
            self.opt.model_name="GrowingNet"
            self.opt.save_root="checkpoints/GrowingNet"
            self.opt.which_iter=1200000
            self.iter[self.opt.model_name] = self.opt.which_iter
            self.opt.check_name="2023-05-11_bust_prev3"
            self.opt.condition=True
            self.opt.num_root = 9000
            self.opt.Bidirectional_growth = Bidirectional_growth
            self.opt.growInv=False
            self.growing_solver = GrowingNetSolver()
            self.growing_solver.initialize(self.opt)
            
            # self.opt.model_name="GrowingNet"
            # self.opt.save_root="checkpoints/GrowingNet"
            # self.opt.which_iter=42240
            # self.iter[self.opt.model_name] = self.opt.which_iter
            # self.opt.check_name="2023-09-04_bust_prev3_hg"
            # self.opt.condition=True
            # self.opt.num_root = 12000
            # self.opt.Bidirectional_growth = Bidirectional_growth
            # self.opt.growInv=False
            # self.growing_solver = GrowingNetSolver()
            # self.growing_solver.initialize(self.opt)
            
            self.opt.model_name=="HairModelingHD"
            self.opt.save_root="checkpoints/HairModelingHD"
            self.opt.blur_ori = False
            self.opt.no_use_depth = True
            self.opt.no_use_pretrain = True
            if use_ori_addinfo:
                self.opt.which_iter=16512 # ori: LocalFilter line 16
                self.opt.check_name="2023-10-26_addori"
            else:
                self.opt.which_iter=295840 # 295840 测试过较好,但有时有空洞  667360:； 700400:加入DB2、DB3以后训练的
                self.opt.check_name="2023-07-31_rot_depth1"
                
            self.opt.use_ori_addinfo=use_ori_addinfo
            self.ModelingHD_solver = HairModelingHDSolver()
            self.ModelingHD_solver.initialize(self.opt)
        else:
            self.opt.voxel_size = "96,128,128"
            if use_hd:
                self.opt.voxel_size = "192,256,256"
            self.opt.model_name="GrowingNet"
            self.opt.save_root="checkpoints/GrowingNet"
            self.opt.which_iter=1200000
            self.iter[self.opt.model_name] = self.opt.which_iter
            self.opt.check_name="2023-05-11_bust_prev3"
            self.opt.condition=True
            self.opt.num_root = 12000
            self.opt.Bidirectional_growth = Bidirectional_growth
            self.opt.growInv=False
            self.growing_solver = GrowingNetSolver()
            self.growing_solver.initialize(self.opt)
            
            self.opt.model_name="HairSpatNet"
            self.opt.save_root="checkpoints/HairSpatNet"
            # self.opt.which_iter=1640000#2023-04-17_bust,1640000 utils 1539 line color5.png，使用color5训练的;input_nc =2
            # self.opt.which_iter=85000#2023-06-06_bust_rot, utils 1539 line body_0.png，使用body_0训练的;input_nc =2
            
            # if use_depth:
            #     self.opt.input_nc = 3
            #     self.opt.check_name="2023-07-31_rot_depth1"
            #     self.opt.which_iter=81840#2023-07-31_rot_depth1,use norm depth;input_nc =3
            # else:
            self.opt.input_nc = 2
            self.opt.which_iter=51000 #2023-07-31_bust_rot, random rot, no depth 39000;input_nc =2  295840:hd训练的略微比51000差，51000
            self.opt.check_name="2023-07-31_bust_rot"
            
            self.opt.no_use_L = True
            self.opt.no_use_depth=True
            self.opt.blur_ori = False
            self.iter[self.opt.model_name] = self.opt.which_iter
            # self.opt.check_name="2023-04-17_bust"
            # self.opt.check_name="2023-06-06_bust_rot"
            self.spat_solver = HairSpatNetSolver()
            self.spat_solver.initialize(self.opt)
        self.sample_num=100
        self.body = trimesh.load_mesh(os.path.join(rFolder,"female_halfbody_medium_join.obj"))
        # opt.model_name=='HairModeling'
        # self.hd_solver=HairModelingHDSolver()
        # self.hd_solver.initialize(opt)
    def eval_ori2dtohair(self):
        import torch
        import numpy as np
        has_orientation = True
        has_ori2d=True
        if has_ori2d:
            pass
            
        if has_orientation:
            occ=torch.load("/data/HairStrand/NeuralHDHairData/DB3/out_occ.pt")#get out_occ.pt from main.py by use gt hair model
            orientation=torch.load("/data/HairStrand/NeuralHDHairData/DB3/out_ori.pt")
            # verts, faces, normals, values = measure.marching_cubes(occ[0,0].cpu().detach().numpy().transpose((2,1,0)), 0.5)
            # verts = transform_Inv(verts,scale=2)
            # hair_mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
            # hair_mesh = trimesh.smoothing.filter_laplacian(hair_mesh, iterations=10)
            # hair_mesh.export(f"/data/HairStrand/NeuralHDHairData/DB3/mesh.obj")
            rgb_image = cv2.imread("/data/HairStrand/NeuralHDHairData/DB3/20231219_203605_manual.png")
            rgb_image=cv2.resize(rgb_image,(640,640))
            orientation = orientation*occ
        
            orientation=orientation.permute(0,2,3,4,1)#[1, 96, 128, 128, 3]
            orientation=orientation.detach().cpu().numpy()
            orientation=orientation * np.array([1, -1, -1])
            orientation=orientation.transpose(0,2,3,4,1)
            _,H,W,C,D=orientation.shape[:]
            orientation=orientation.reshape(H ,W,C*D)
        m=np.identity(3)
        color = [0,0,0,255]
        points_same,points,segments,colors = self.growing_solver.inference(orientation,m,hair_img=rgb_image,avg_color=color, sample_num=self.sample_num)
        # mask = np.sum(colors,axis=1)
        # colors[np.where(mask==255)]=color
        sample_num=self.sample_num
        start = segments[0]
        segments = np.cumsum(segments)
        segments = np.insert(segments,0,0)[:-1]
        if self.delete_far:
            connect=False
            points_same,points,segments = self.froot.getNewRoot(points_same.reshape((-1,self.sample_num,3)),points,segments,connect=connect)
            points_same = np.array(points_same).reshape([-1,3])
            if connect:
                sample_num=self.sample_num+1
        use_unity = False
        if use_unity:
            import base64
            # points=np.load("/media/yxh/My Passport/ths/NeuralHDHairData/DB3/DB3.npy")
            nums= 100
            # points=pyBsplineInterp.GetBsplineInterp(points,segments,nums,3)
            # transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.31],dimensionality=3)
            m = transform.SimilarityTransform(scale=[7.3,8.5,8],translation=[-0.0,-14.63,-0],dimensionality=3)#DB3
            # m = transform.SimilarityTransform(scale=[7.6,8.5,8],translation=[0.03,-14.63,-0],dimensionality=3)#DB2
            points = np.array(points).reshape([-1,3])
            points=transform.matrix_transform(points,m.params).astype('float32')

            # segments_same = np.array(range(0,len(points)//nums))*nums
            bytes_array = points.tobytes() 
            points_str = base64.b64encode(bytes_array).decode('utf-8') 

            params = {"reqCode":"female_1","points":points_str,"segments":segments.tolist(),"colors":[[34,36,45,255]],"version":0.6}


            x=json.dumps(params)
            data = {
                "method": "CreateHair",
                "params": x
            }
            # writejson("test.json",data)
            ip_port="http://10.10.53.18:3889"
            ret = requests.post(ip_port,data=json.dumps(data))
            print(ret)
        segments_same = np.array(range(0,len(points_same)//sample_num))*sample_num # 固定100个点
        np.save("x.npy",points_same)
        # writejson("test1.json",{"points":points.reshape((-1,3)).tolist(),"segments":segments.tolist()})
        return points.reshape((-1,3)),segments,colors
    @timeCost
    def inference(self,image,gender="" ,name="",save_path="",use_gt=False,use_unity=False,use_NeuralHaircut=True):
        self.opt.test_file = name.split('.')[0]
        if use_unity:
            reset()
        # set_camera()
        logging.info("enter strand2d")
        if  self.HairFilterLocal:
            ori2D,bust,color,rgb_image,revert_rot,cam_intri,cam_extri = self.img_filter.pyfilter2neuralhd(image,gender,name,use_gt=use_gt)
        else:
            imgB64 = cvmat2base64(image)
            ori2D,bust,color,rgb_image,revert_rot,cam_intri,cam_extri = self.img_filter.request_HairFilter(name,'img',imgB64)
        logging.info("leave strand2d,enter strand3d")
        # kernel = np.ones((3,3),np.uint8)
        # ori2D = cv2.erode(ori2D,kernel,iterations=1)
        debug=True
        if debug:
            cv2.imwrite(f"{self.opt.test_file}_ori.png",ori2D)
            cv2.imwrite(f"{self.opt.test_file}_bust.png",(bust*255).astype('uint8'))
            cv2.imwrite(f"{self.opt.test_file}_rgb.png",rgb_image)
            cv2.imwrite(f"{self.opt.test_file}_color.png",color)
            np.save(f"{self.opt.test_file}_revert_rot.npy",revert_rot)
            np.save(f"{self.opt.test_file}_cam_intri.npy",cam_intri)
            np.save(f"{self.opt.test_file}_cam_extri.npy",cam_extri)
        depth_norm = None
        if self.opt.input_nc==3 or self.use_depth:
            if not self.use_strand:
                mask = np.zeros((ori2D.shape[0],ori2D.shape[1]))
                parse = ori2D[:, :, 2]
                mask1=ori2D[:, :, [0, 1]]
                mask1[np.where(parse<0.8)]=[0,0]
                mask[np.where(np.sum(mask1,axis=2)>0)]=1
                # cv2.imshow("4",mask)
                # cv2.waitKey()
            else:
                mask = np.zeros((ori2D.shape[0],ori2D.shape[1]))
                mask1=ori2D[:, :, [1, 2]]
                # mask1[np.where(parse<0.8)]=[0,0]
                mask[np.where(np.sum(mask1,axis=2)>0)]=1
                # cv2.imshow("4",mask)
                # cv2.waitKey()
            if  self.HairFilterLocal:
                depth_norm=self.img_filter.get_depth(rgb_image,mask)
            else:
                rgbB64 = cvmat2base64(rgb_image)
                maskB64 = cvmat2base64(mask)
                depth_norm = self.img_filter.request_depth(name,'depth',rgbB64,maskB64)#dtype('float64')
        if not self.use_modeling:
            if self.opt.input_nc==3:
                orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=None,norm_depth=depth_norm,use_bust=False,name=self.opt.test_file)
            # ori2D = image
            else:
                if self.use_hd:
                    orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=bust,name=self.opt.test_file,resolution=[192,256,256])#(128, 128, 288)
                else:
                    orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=bust,name=self.opt.test_file)
        else:
            orientation,out_occ = self.ModelingHD_solver.inference(ori2D,bust=bust,norm_depth=depth_norm,name=name)#orientation:128*128*3*96
            if self.get_cartoon:
                verts, faces, normals, values = measure.marching_cubes(out_occ[0,0].cpu().numpy().transpose((2,1,0)), 0.5)
                verts = transform_Inv(verts,scale=2)
                verts = verts+np.array([0.00703544,-1.58652416,-0.01121912])
                verts = np.dot(verts, revert_rot)+np.array([-0.00703544,1.58652416,0.01121912])
                hair_mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
                hair_mesh = trimesh.smoothing.filter_laplacian(hair_mesh, iterations=10)
                hair_mesh.export(f"{self.opt.test_file}.obj")
                _,_,rgb = render_cartoon(hair_mesh,self.body,mesh_colors=np.array([177, 177, 177, 255]))
                cv2.imwrite(f"{self.opt.test_file}.png",rgb)
                return verts,faces,normals
        np.save(f"{self.opt.test_file}_orientation.npy", orientation)
        np.save(f"{self.opt.test_file}_occ.npy", out_occ.cpu().numpy())
        use_NeuralHaircut = False
        if use_NeuralHaircut:
            import torch
            # for debug growing net
            ori2D = cv2.imread(f"{self.opt.test_file}_ori.png")
            # cv2.imwrite(f"{self.opt.test_file}_bust.png",(bust*255).astype('uint8'))
            rgb_image = cv2.imread(f"{self.opt.test_file}_rgb.png")
            # cv2.imwrite(f"{self.opt.test_file}_color.png",color)
            revert_rot = np.load(f"{self.opt.test_file}_revert_rot.npy")
            cam_intri = np.load(f"{self.opt.test_file}_cam_intri.npy")
            cam_extri = np.load(f"{self.opt.test_file}_cam_extri.npy")
            ori=np.load(f"{self.opt.test_file}_orientation.npy")
            out_occ = np.load(f"{self.opt.test_file}_occ.npy")
            
            from NeuralHaircut.run_strands_optimization import Runner
            runner = Runner("./configs/monocular/neural_strands_w_camera_fitted.yaml", "person_0","monocular",hair_conf_path="./configs/hair_strands_textured.yaml", exp_name="second_stage_person_0")
            image = torch.from_numpy(rgb_image)
            mask = cv2.cvtColor(rgb_image,cv2.RGB2GRAY)
            mask[mask>0]=255
            #ori:128*128*3*96
            ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
            ori = ori.transpose([0, 1, 3, 2])# ori: 128*128*96*3

            transfer = True
            ori = np.ascontiguousarray(ori)
            s = [ori.shape[0],ori.shape[1],ori.shape[2]]
            scale=1
            if s[0]==256:
                scale=2
            # 旋转方向场到正面
            ori = np.transpose(ori, (1,0,2,3)) #ori :交换x,y轴到 128,128,96,3
            ori = ori[::-1, :, :,  :]   #x轴flip已对应旋转矩阵方向
            
            from skimage import transform as trans
            mask=np.linalg.norm(ori,axis=-1)#ori : 128,128,96,3
            gt_occ=(mask>0).astype(np.float32)
            mask1 = np.array(np.where(gt_occ>0))
            gt_occ1=mask1.T-np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])

            new_gt_occ = np.dot(gt_occ1, revert_rot)+np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])
            new_gt_occ = new_gt_occ.T.astype('int')
            
            index = (new_gt_occ[2] >= 0) & (new_gt_occ[2] <= s[2]-1)&(new_gt_occ[0] >= 0) & (new_gt_occ[0] <= s[0]-1)&(new_gt_occ[1] >= 0) & (new_gt_occ[1] <= s[1]-1)
            new_gt_occ = new_gt_occ[:,index]
            mask1 = mask1[:,index]
            ori1 = ori[tuple(mask1)]
            new_ori1 = np.dot(ori1.reshape((-1,3)),revert_rot)
            ori = np.zeros_like(ori)
            ori[new_gt_occ[0],new_gt_occ[1],new_gt_occ[2]] = new_ori1
            
            ori = ori[::-1, :, :,  :]
            ori = np.transpose(ori, (1,0,2,3)) 
            
            ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
            ori = ori.transpose([0, 1, 3, 2])
            mask=np.linalg.norm(ori.reshape((ori.shape[0], ori.shape[1], -1, 3)),axis=-1)
            gt_occ=(mask>0).astype(np.float32)
            runner.train(image,mask,ori2D,ori,gt_occ,cam_extri,cam_intri)

        if not isinstance(orientation,np.ndarray) and orientation==None:
            return
        # for debug growing net
        # import scipy
        # orientation = scipy.io.loadmat("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/DB1/Ori_gt.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
        # orientation = np.load(os.path.join("/media/yxh/My Passport/ths/neuraldata1","DB1/Ori_gt_hg.npy"))
        # rgb_image = np.zeros((256,256,3))
        # color = np.array([255,0,0,255])
        # m = np.identity(3)
        # draw_arrows_by_projection1(os.path.join("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/","DB1"),self.iter["GrowingNet"],draw_occ=True,hair_ori=orientation)
        m=revert_rot
        points_same,points,segments,colors = self.growing_solver.inference(orientation,m,hair_img=rgb_image,avg_color=color,sample_num=self.sample_num)
        mask = np.sum(colors,axis=1)
        colors[np.where(mask==255)]=color
        #旋转点
        # points = points+np.array([0.00703544,-1.58652416,-0.01121912])
        
        # points = np.dot(points, m)+np.array([-0.00703544,1.58652416,0.01121912])
        
        # write_data(os.path.join(save_path,f"{name.split('.')[0]}.data"),points,segments)
        # write_data(os.path.join(save_path,f"{name.split('.')[0]}.cin"),points,segments,colors)
        # points,segments,colors = get_data(os.path.join(save_path,f"{name.split('.')[0]}.cin"),has_color=True)
        # 采样点
        # points = process_list(points,segments,self.sample_num)
        sample_num=self.sample_num
        start = segments[0]
        segments = np.cumsum(segments)
        segments = np.insert(segments,0,0)[:-1]
        if self.delete_far:
            connect=False
            points_same,points,segments = self.froot.getNewRoot(points_same.reshape((-1,self.sample_num,3)),points,segments,connect=connect)
            points_same = np.array(points_same).reshape([-1,3])
            if connect:
                sample_num=self.sample_num+1
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        # m=[]
        # _,bust,img2 = render_strand(points,segments,self.body,width=512,vertex_colors=np.array([127, 127, 127, 255]),strand_color=colors,orientation=[],intensity=3,matrix=m,mask=False)
        # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_py.png"),img2)
        # colors[:] = color
        # _,bust,img2 = render_strand(points,segments,self.body,width=512,vertex_colors=np.array([127, 127, 127, 255]),strand_color=colors,orientation=[],intensity=3,matrix=m,mask=False)
        # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_py1.png"),img2)
        if use_unity:
            # 转换到unity空间
            # 最初版的female_halfbody_medium.obj对齐到unity人脸模型
            # m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反，
            # v0.7版本的female_halfbody_medium.obj对齐到unity人脸模型
            m = transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.31],dimensionality=3)
            points = np.array(points).reshape([-1,3])
            points=transform.matrix_transform(points,m.params)
            # writejson("test.json",{"points":points.reshape((-1,100,3))[:100].tolist()})
            trans_hair(points,segments,color[[2,1,0,3]].tolist(),sample_num)
            # # # time.sleep(1)
            if use_gt:
                set_camera(flag=0)
                render(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
                time.sleep(1)
                img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
                img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"),img)
                set_camera(flag=1)
                render(os.path.join(save_path,f"{name.split('.')[0]}_2g.png"))
                time.sleep(1)
                img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_2g.png"))
                img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_2g.png"),img)
                set_camera(-30)
                render(os.path.join(save_path,f"{name.split('.')[0]}_3g.png"))
                time.sleep(1)
                img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_3g.png"))
                img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_3g.png"),img)
            else:
                set_camera(flag=0)
                save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out_paper/"
                render(os.path.join(save_path,f"{name.split('.')[0]}_1.png"))
                time.sleep(1)
                # img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1.png"))
                # img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1.png"),img)
                set_camera(flag=1)
                render(os.path.join(save_path,f"{name.split('.')[0]}_2.png"))
                time.sleep(1)
                # img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_2.png"))
                # img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_2.png"),img)
                set_camera(flag=2)
                render(os.path.join(save_path,f"{name.split('.')[0]}_3.png"))
                time.sleep(1)
                # img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_3.png"))
                # img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
                # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_3.png"),img)
        segments_same = np.array(range(0,len(points_same)//sample_num))*sample_num # 固定100个点
        # writejson("test1.json",{"points":points.reshape((-1,3)).tolist(),"segments":segments.tolist()})
        return points.reshape((-1,3)),segments,colors
        
if __name__=="__main__":
    # test marching_cubes
    # body = trimesh.load_mesh("./female_halfbody_medium_join.obj")
    # ori=np.load(os.path.join("/data/HairStrand/NeuralHDHairData","strands00001",'Ori_gt_hg.npy'),allow_pickle=True)
    # mask=np.linalg.norm(ori.reshape((ori.shape[0], ori.shape[1], -1, 3)),axis=-1)
    # gt_occ=(mask>0).astype(np.float32)
    # gt_occ[gt_occ>=0.5]=1
    # gt_occ[gt_occ<0.5]=0
    # verts, faces, normals, values = measure.marching_cubes(gt_occ.transpose((1,0,2)), 0.5)
    # verts = transform_Inv(verts,scale=ori.shape[0]//128)
    # hair_mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
    # hair_mesh = trimesh.smoothing.filter_laplacian(hair_mesh, iterations=10)
    # hair_mesh.export(f"strands00001.obj")
    # _,_,rgb = render_cartoon(hair_mesh,body,mesh_colors=np.array([177, 177, 177, 255]))
    # cv2.imwrite(f"strands00001.png",rgb[...,:3])
    
    use_unity=False
    gender = ['female','male']
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if use_unity:
        set_bgcolor()
        set_camera(flag=1)
    hair_infe = strand_inference(os.path.dirname(os.path.dirname(__file__)),HairFilterLocal=True,use_modeling=True,use_step=True,\
                                 use_strand=True,Bidirectional_growth=True,gpu_ids=[2],get_cartoon=False)
    # hair_infe.eval_ori2dtohair()
    save_path = os.path.join(os.path.dirname(__file__),"../data/test/out_paper")
    for g in gender:
        test_dir = os.path.join(os.path.dirname(__file__),"../data/test/paper")
        file_names = os.listdir(test_dir)
        file_names = ['乔小刀.jpg']#,'img_0044.png'
        for name in tqdm(file_names[:]):#31:32，19
            # name = "female_20.jpg"
            test_file = os.path.join(test_dir,name)
            img = cv2.imread(test_file)
            cv2.imwrite(os.path.join(save_path, name),img)
            hair_infe.inference(img,g,name,save_path,use_gt=False,use_unity=use_unity)
