# 打开realistic-exe-linux项目可执行文件进行渲染
from options.inference_options import InferenceOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
from Tools.to_unity import *
from image_filter import filter_crop
import cv2

import time
from tqdm import tqdm
from Tools.file_io import *
from skimage import transform as trans
from dataload.render_strand import render_strand
class strand_inference:
    def __init__(self,rFolder,use_hd=False,use_step=True,use_depth=False,use_strand=False,Bidirectional_growth=False) -> None:
        """_summary_

        Args:
            rFolder (_type_): _description_
            use_step (bool, optional): 网络直接生成分割图和方向图，151 docker restart segmentall-server. Defaults to True.
            use_depth (bool, optional): 需要使用归一化后的头发深度图作为输入. Defaults to False.
            use_strand (bool, optional): segmentanything网络直接生成分割图,本地生成方向图，151 docker restart segmentall-server. Defaults to False.
            Bidirectional_growth (bool, optional): _description_. Defaults to False.
        """   
        self.use_strand = use_strand
        if use_hd:
            use_depth=True     
        self.img_filter = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code",\
                                      "/home/yxh/Documents/company/NeuralHDHair/data/test",\
                                      use_step=use_step,use_depth=use_depth,use_strand=use_strand)
        # image = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/test/image.jpg")
        # image = cv2.resize(image,(640,640))
        # ori2D,mask = self.img_filter.pyfilter2neuralhd(image)
        self.use_step=use_step
        self.opt=InferenceOptions().initialize(use_hd=use_hd)
        self.iter = {}
        self.froot = find_root(rFolder)
        self.opt.gpu_ids=[0]
        gpu_str = [str(i) for i in self.opt.gpu_ids]
        gpu_str = ','.join(gpu_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        self.opt.current_path = rFolder
        self.opt.name="2023-05-06_bust_test"
        self.opt.save_dir="data/Train_input"
        self.opt.is_Train = False
        self.opt.voxel_size = "96,128,128"
        self.use_hd = use_hd
        if use_hd:
            self.opt.voxel_size = "192,256,256"
        # self.opt.model_name="GrowingNet"
        # self.opt.save_root="checkpoints/GrowingNet"
        # self.opt.which_iter=1200000
        # self.iter[self.opt.model_name] = self.opt.which_iter
        # self.opt.check_name="2023-05-11_bust_prev3"
        # self.opt.condition=True
        # self.opt.num_root = 12000
        # self.opt.Bidirectional_growth = Bidirectional_growth
        # self.opt.growInv=False
        # self.growing_solver = GrowingNetSolver()
        # self.growing_solver.initialize(self.opt)
        if use_hd:
            self.opt.model_name=="HairModelingHD"
            self.opt.save_root="checkpoints/HairModelingHD"
            self.opt.blur_ori = False
            self.opt.no_use_depth = True
            
            self.opt.pretrain_path="2023-07-31_rot_depth1/checkpoint/HairModelingGlobal_25456.pth"
            self.opt.which_iter=25456
            self.opt.check_name="2023-07-31_rot_depth1"
            self.ModelingHD_solver = HairModelingHDSolver()
            self.ModelingHD_solver.initialize(self.opt)
        else:
            self.opt.model_name="HairSpatNet"
            self.opt.save_root="checkpoints/HairSpatNet"
            # self.opt.which_iter=1640000#2023-04-17_bust,1640000 utils 1539 line color5.png，使用color5训练的;input_nc =2
            # self.opt.which_iter=885000#2023-06-06_bust_rot, utils 1539 line body_0.png，使用body_0训练的;input_nc =2
            
            if use_depth:
                self.opt.input_nc = 3
                self.opt.check_name="2023-07-31_rot_depth1"
                self.opt.which_iter=81840#2023-07-31_rot_depth1,use norm depth;input_nc =3
            else:
                self.opt.input_nc = 2
                self.opt.which_iter=51000#2023-07-31_bust_rot, random rot, no depth 39000;input_nc =2
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
        self.body = trimesh.load_mesh(os.path.join(rFolder,"female_halfbody_medium.obj"))
        # opt.model_name=='HairModeling'
        # self.hd_solver=HairModelingHDSolver()
        # self.hd_solver.initialize(opt)
    @timeCost
    def inference(self,image,gender="" ,name="",save_path="",use_gt=False):
        self.opt.test_file = name.split('.')[0]
        if use_unity:
            reset()
        # set_camera()
        ori2D,bust,color,rgb_image = self.img_filter.pyfilter2neuralhd(image,gender,name,use_gt=use_gt)
        kernel = np.ones((3,3),np.uint8)
        ori2D = cv2.erode(ori2D,kernel,iterations=1)
        cv2.imshow('1',ori2D)
        cv2.imshow('2',bust)
        cv2.imshow('3',rgb_image)
        cv2.waitKey()
        if self.opt.input_nc==3 or self.use_hd:
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
            depth_norm=self.img_filter.get_depth(rgb_image,mask)
        if not self.use_hd:
            if self.opt.input_nc==3:
                orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=None,norm_depth=depth_norm,use_bust=False,name=name)
            # ori2D = image
            else:
                orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=bust,name=name)#(128, 128, 288)
        else:
            orientation = self.ModelingHD_solver.inference(ori2D,use_step=self.use_step,bust=bust,norm_depth=depth_norm,name=name)
        if not isinstance(orientation,np.ndarray) and orientation==None:
            return
        import scipy
        orientation = scipy.io.loadmat("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/DB1/Ori_gt.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
        # orientation = np.load(os.path.join("/media/yxh/My Passport/ths/neuraldata1","DB1/Ori_gt_hg.npy"))
        
        rgb_image = np.zeros((256,256,3))
        color = np.array([255,0,0,255])
        m = np.identity(3)
        # draw_arrows_by_projection1(os.path.join("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/","DB1"),self.iter["GrowingNet"],draw_occ=True,hair_ori=orientation)
        # m=self.img_filter.revert_rot
        points,segments,colors = self.growing_solver.inference(orientation,m,hair_img=rgb_image,avg_color=color,sample_num=self.sample_num)
        mask = np.sum(colors,axis=1)
        colors[np.where(mask==255)]=color
        #旋转点
        # points = points+np.array([0.00703544,-1.58652416,-0.01121912])
        
        # points = np.dot(points, m)+np.array([-0.00703544,1.58652416,0.01121912])
        # save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out/"
        write_data(os.path.join(save_path,f"{name.split('.')[0]}.data"),points,segments)
        write_data(os.path.join(save_path,f"{name.split('.')[0]}.cin"),points,segments,colors)
        # points,segments,colors = get_data(os.path.join(save_path,f"{name.split('.')[0]}.cin"),has_color=True)
        # 采样点
        # points = process_list(points,segments,self.sample_num)
        # points = self.froot.getNewRoot(points.reshape((-1,self.sample_num,3)))
        # sample_num=self.sample_num+1
        sample_num=self.sample_num
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        # m=[]
        # _,bust,img2 = render_strand(points,segments,self.body,width=512,vertex_colors=np.array([127, 127, 127, 255]),strand_color=colors,orientation=[],intensity=3,matrix=m,mask=False)
        # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_py.png"),img2)
        # colors[:] = color
        # _,bust,img2 = render_strand(points,segments,self.body,width=512,vertex_colors=np.array([127, 127, 127, 255]),strand_color=colors,orientation=[],intensity=3,matrix=m,mask=False)
        # cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_py1.png"),img2)
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
            render(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
            time.sleep(1)
            img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
            img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
            cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"),img)
            set_camera(30)
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
            render(os.path.join(save_path,f"{name.split('.')[0]}_1.png"))
            time.sleep(1)
            img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1.png"))
            img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
            cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1.png"),img)
            set_camera(30)
            render(os.path.join(save_path,f"{name.split('.')[0]}_2.png"))
            time.sleep(1)
            img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_2.png"))
            img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
            cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_2.png"),img)
            set_camera(-30)
            render(os.path.join(save_path,f"{name.split('.')[0]}_3.png"))
            time.sleep(1)
            img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_3.png"))
            img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
            cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_3.png"),img)
        
if __name__=="__main__":
    # gt=cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/strand_map/10_f.png")#R:（0,1）表示（向右，向左）；G：第二通道，（0,1）表示（向下，向上）
    # image = cv2.imread(f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/img/1a6104485fb61f2371560063b7a15738.png")
    # img_filter = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code","/home/yxh/Documents/company/NeuralHDHair/data/test",use_step=True)
    # ori2D,_ = img_filter.pyfilter2neuralhd(image,"female","10_f",use_gt=False)
    # # TODO:两种方式得到的segment图不太一样，seg中的对散发也能分割。哪个比较好 后续进行实验
    # cv2.imshow("1",gt)
    # cv2.waitKey()
    # reset()
    # segments,points = readhair("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/strands00001/hair_delete.hair")
    # import pickle
    # import scipy
    # # # Load the pickle file
    # with open(os.path.join("/home/yxh/Documents/HairNet/hairstylepickles",'strands00356.pkl'), 'rb') as f:
    #     points = pickle.load(f)
    # # hair = scipy.io.loadmat("/home/yxh/Documents/company/NeuralHDHair/roots1.mat")
    # # points = hair["points"].reshape((-1,9,3))
    # start = 0
    # roots = []
    # for i in range(0,len(points)):
    #     if len(points[i])>1:
    #         new_root = np.array(points[i])[0:1]
    #         roots.append(new_root)
    # roots = np.array(roots)
    # roots = roots.reshape((-1,3))
    # # scipy.io.savemat("roots1.mat", {"roots":roots})
    # m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
    # roots = transform.matrix_transform(roots,m.params)
    # trans_hair(roots,2)  
    use_unity=False
    gender = ['female','male']
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if use_unity:
        set_bgcolor()
    hair_infe = strand_inference(os.path.dirname(os.path.dirname(__file__)),use_hd=True,use_step=False,use_depth=False,use_strand=True,Bidirectional_growth=True)
    save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out_paper/"
    for g in gender:
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/test/{g}"
        # test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/img"
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/test/paper"
        file_names = os.listdir(test_dir)
        for name in tqdm(file_names[4:]):#31:32，19
            # name = "10_f.png"
            test_file = os.path.join(test_dir,name)
            img = cv2.imread(test_file)
            cv2.imwrite(os.path.join(save_path, name),img)
            hair_infe.inference(img,g,name,save_path,use_gt=False)