from options.inference_options import InferenceOptions
from solver.StepNetSolver import StepNetSolver
import os
from image_filter import filter_crop
import cv2
from Tools.resample import resample,process_list
import time
from tqdm import tqdm
class step_inference:
    def __init__(self,rFolder) -> None:
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
        
        opt.check_name="2023-06-27"
        opt.save_dir="data/Train_input"
        opt.is_Train = False
        opt.model_name="StepNet"
        opt.save_root="checkpoints/StepNet"
        opt.which_iter=1135008
        self.iter[opt.model_name] = opt.which_iter
        self.step_solver = StepNetSolver()
        self.step_solver.initialize(opt)
        
    def inference(self,image,name="",use_gt=False):
        orientation = self.step_solver.inference(image)
        
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        # save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out/"
        # if use_gt:
        #     img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
        #     img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
        #     cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"),img)
        # else:
        #     img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1.png"))
        #     img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
        #     cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1.png"),img)
        
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
    hair_infe = step_inference(os.path.dirname(os.path.dirname(__file__)))
    save_path = "/home/yxh/Documents/company/NeuralHDHair/data/test/out/"
    for g in gender:
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/test/{g}"
        test_dir = f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/img"
        file_names = os.listdir(test_dir)
        for name in tqdm(file_names[:]):#:32
            # name = "10_f.png"
            test_file = os.path.join(test_dir,name)
            img = cv2.imread(test_file)
            cv2.imwrite(os.path.join(save_path, name),img)
            hair_infe.inference(img,name,use_gt=False)