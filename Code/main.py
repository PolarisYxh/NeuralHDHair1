from Tools.iter_counter import IterationCounter
from Tools.visualizer import Visualizer
from options.train_options import TrainOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
from solver.StepNetSolver import StepNetSolver
from dataload import data_loader
import os
if __name__ == '__main__':
    opt=TrainOptions().parse()
    gpu_str=[str(i) for i in opt.gpu_ids]
    gpu_str=','.join(gpu_str)
    os.environ['CUDA_VISIBLE_DEVICES'] =gpu_str
    iter_counter = IterationCounter(opt, 0)
    visualizer=Visualizer(opt)
    dataloader=data_loader(opt)
    if opt.model_name == "GrowingNet":
        g_slover = GrowingNetSolver()
    elif opt.model_name == "HairSpatNet":
        g_slover = HairSpatNetSolver()
    elif opt.model_name=='HairModelingHD':
        g_slover=HairModelingHDSolver()
    elif opt.model_name=="StepNet":
        # opt.isTrain=True
        test_dataloader=data_loader(opt)
        g_slover=StepNetSolver()
        # opt.isTrain=True
    g_slover.initialize(opt)
    if opt.model_name=="StepNet":
        g_slover.train(iter_counter,dataloader,test_dataloader,visualizer)
    else:
        g_slover.train(iter_counter,dataloader,visualizer)
