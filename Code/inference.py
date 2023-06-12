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
class strand_inference:
    def __init__(self,rFolder) -> None:
        self.img_filter = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code","/home/yxh/Documents/company/NeuralHDHair/data/test")
        # image = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/test/image.jpg")
        # image = cv2.resize(image,(640,640))
        # ori2D,mask = self.img_filter.pyfilter2neuralhd(image)
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
        opt.which_iter=690000
        self.iter[opt.model_name] = opt.which_iter
        opt.check_name="2023-05-11_bust_prev3"
        opt.condition=True
        opt.num_root = 10000
        opt.Bidirectional_growth = True
        opt.growInv=False
        self.growing_solver = GrowingNetSolver()
        self.growing_solver.initialize(opt)
        
        opt.model_name="HairSpatNet"
        opt.save_root="checkpoints/HairSpatNet"
        opt.which_iter=1640000#2023-04-17_bust,1640000 utils 1505 line color5.png
        # opt.which_iter=665000#2023-05-25_bust
        opt.no_use_L = True
        opt.no_use_depth=True
        opt.blur_ori = True
        self.iter[opt.model_name] = opt.which_iter
        opt.check_name="2023-04-17_bust"
        # opt.check_name="2023-05-25_bust"
        self.spat_solver = HairSpatNetSolver()
        self.spat_solver.initialize(opt)
        
        # opt.model_name=='HairModeling'
        # self.hd_solver=HairModelingHDSolver()
        # self.hd_solver.initialize(opt)
        
    def inference(self,image,gender="" ,name=""):
        ori2D,_ = self.img_filter.pyfilter2neuralhd(image,gender,name)
        # ori2D = image
        orientation = self.spat_solver.inference(ori2D)
        # draw_arrows_by_projection1(os.path.join("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/","DB1"),self.iter["GrowingNet"],draw_occ=True,hair_ori=orientation)

        points,segments = self.growing_solver.inference(orientation)
        # 打开realistic-exe-linux项目可执行文件进行渲染
        reset()
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # points,segments=resample(points,segments)
        points = process_list(points,segments)
        points = np.array(points).reshape([-1,3])
        trans_hair(points,segments)
        time.sleep(1)
        render(f"/home/yxh/Documents/company/NeuralHDHair/data/test/{name.split('.')[0]}_2.png")
        
if __name__=="__main__":
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
    for g in gender:
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/test/{g}"
        file_names = os.listdir(test_dir)
        for name in tqdm(file_names[2:]):
            # name = "Screenshot from 2023-03-15 15-36-32_f.png"
            test_file = os.path.join(test_dir,name)
            img = cv2.imread(test_file)
            hair_infe = strand_inference(os.path.dirname(os.path.dirname(__file__)))
            hair_infe.inference(img,g,name)