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
from Tools.resample import resample,process_list
import time
from tqdm import tqdm
from Tools.file_io import *
from skimage import transform as trans
class strand_inference:
    def __init__(self,rFolder,use_step,Bidirectional_growth=False) -> None:
        self.img_filter = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code","/home/yxh/Documents/company/NeuralHDHair/data/test",use_step=use_step)
        # image = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/test/image.jpg")
        # image = cv2.resize(image,(640,640))
        # ori2D,mask = self.img_filter.pyfilter2neuralhd(image)
        self.use_step=use_step
        opt=InferenceOptions().initialize()
        self.iter = {}
        opt.gpu_ids=[0]
        gpu_str = [str(i) for i in opt.gpu_ids]
        gpu_str = ','.join(gpu_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        opt.current_path = rFolder
        opt.name="2023-05-06_bust_test"
        opt.save_dir="data/Train_input"
        opt.is_Train = False
        
        opt.model_name="GrowingNet"
        opt.save_root="checkpoints/GrowingNet"
        opt.which_iter=1200000
        self.iter[opt.model_name] = opt.which_iter
        opt.check_name="2023-05-11_bust_prev3"
        opt.condition=True
        opt.num_root = 7500
        opt.Bidirectional_growth = Bidirectional_growth
        opt.growInv=False
        self.growing_solver = GrowingNetSolver()
        self.growing_solver.initialize(opt)
        
        opt.model_name="HairSpatNet"
        opt.save_root="checkpoints/HairSpatNet"
        # opt.which_iter=1640000#2023-04-17_bust,1640000 utils 1539 line color5.png
        opt.which_iter=665000#2023-05-25_bust, utils 1539 line body_0.png
        opt.no_use_L = True
        opt.no_use_depth=True
        opt.blur_ori = True
        self.iter[opt.model_name] = opt.which_iter
        # opt.check_name="2023-04-17_bust"
        opt.check_name="2023-05-25_bust"
        self.spat_solver = HairSpatNetSolver()
        self.spat_solver.initialize(opt)
        self.sample_num=100
        # opt.model_name=='HairModeling'
        # self.hd_solver=HairModelingHDSolver()
        # self.hd_solver.initialize(opt)
        
    def inference(self,image,gender="" ,name="",use_gt=False):
        reset()
        # set_camera()
        ori2D,bust = self.img_filter.pyfilter2neuralhd(image,gender,name,use_gt=use_gt)
        # ori2D = image
        orientation = self.spat_solver.inference(ori2D,use_step=self.use_step,bust=bust)
        # draw_arrows_by_projection1(os.path.join("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/","DB1"),self.iter["GrowingNet"],draw_occ=True,hair_ori=orientation)

        points,segments = self.growing_solver.inference(orientation)
        #旋转点
        points = points+np.array([0.00703544,-1.58652416,-0.01121912])
        m=self.img_filter.revert_rot
        points = np.dot(points, m)+np.array([-0.00703544,1.58652416,0.01121912])

        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # points,segments=resample(points,segments)
        points = process_list(points,segments,self.sample_num)
        points = np.array(points).reshape([-1,3])
        # points=points.reshape((-1,self.sample_num,3))
        # points=points[:,:2,:].reshape((-1,3))
        # self.sample_num=2
        trans_hair(points,segments,self.sample_num)
        time.sleep(1)
        save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out/"
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
    gender = ['female','male']
    set_bgcolor()
    hair_infe = strand_inference(os.path.dirname(os.path.dirname(__file__)),use_step=True,Bidirectional_growth=True)
    save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out/"
    for g in gender:
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/test/{g}"
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/img"
        file_names = os.listdir(test_dir)
        for name in tqdm(file_names[0:]):#:32
            # name = "10_f.png"
            test_file = os.path.join(test_dir,name)
            img = cv2.imread(test_file)
            cv2.imwrite(os.path.join(save_path, name),img)
            hair_infe.inference(img,g,name,use_gt=False)