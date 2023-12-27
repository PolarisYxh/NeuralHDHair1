from Tools.iter_counter import IterationCounter
from Tools.visualizer import Visualizer
from options.train_options import TrainOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
from solver.HairModelingHDCalSolver import HairModelingHDCalSolver
from solver.StepNetSolver import StepNetSolver
from solver.OriginStepNetSolver import OriginStepNetSolver
from dataload import data_loader
import os
import platform
if __name__ == '__main__':
    opt=TrainOptions().parse()
    gpu_str=[str(i) for i in opt.gpu_ids]
    gpu_str=','.join(gpu_str)
    os.environ['CUDA_VISIBLE_DEVICES'] =gpu_str
    plat = platform.system().lower()
    if plat != 'windows':
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['EGL_DEVICE_ID']=str(opt.gpu_ids[0])
    iter_counter = IterationCounter(opt, 0)
    visualizer=Visualizer(opt)
    dataloader=data_loader(opt)
    if opt.model_name == "GrowingNet":
        g_slover = GrowingNetSolver()
    elif opt.model_name == "HairSpatNet":
        opt.isTrain=False
        test_dataloader=data_loader(opt)
        opt.isTrain=True
        g_slover = HairSpatNetSolver()
    elif opt.model_name=='HairModelingHD':
        g_slover=HairModelingHDSolver()
    elif opt.model_name=='HairModelingHDCal':
        g_slover=HairModelingHDCalSolver()
    elif opt.model_name=="StepNet":
        opt.isTrain=False
        test_dataloader=data_loader(opt)
        g_slover=StepNetSolver()
        opt.isTrain=True
    elif opt.model_name=="OriginStepNet":
        opt.isTrain=False
        test_dataloader=data_loader(opt)
        g_slover=OriginStepNetSolver()
        opt.isTrain=True
    g_slover.initialize(opt)
    if opt.model_name=="StepNet" or opt.model_name=="HairSpatNet" or opt.model_name=="OriginStepNet":
        g_slover.train(iter_counter,dataloader,test_dataloader,visualizer)
    else:
        g_slover.train(iter_counter,dataloader,visualizer)
