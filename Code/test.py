from options.test_options import TestOptions
from dataload import data_loader
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
from solver.StepNetSolver import StepNetSolver
import os
opt=TestOptions().parse()
gpu_str = [str(i) for i in opt.gpu_ids]
gpu_str = ','.join(gpu_str)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
dataloader=data_loader(opt)
if opt.model_name=="GrowingNet":
    g_slover = GrowingNetSolver()
elif opt.model_name=="HairSpatNet":
    g_slover = HairSpatNetSolver()
elif opt.model_name=='HairModeling':
    g_slover=HairModelingHDSolver()
elif opt.model_name=="StepNet":
    opt.name="2023-05-06_bust_test"
    opt.check_name="2023-06-27"
    opt.save_dir="data/Train_input1"
    opt.model_name="StepNet"
    opt.save_root="checkpoints/StepNet"
    opt.which_iter=1135008 #0.070761077
    # opt.which_iter=15000 0.07236522436141968
    # opt.which_iter=5004 #0.07095076888799667
    opt.isTrain=False
    test_dataloader=data_loader(opt)
    g_slover=StepNetSolver()

g_slover.initialize(opt)
g_slover.test(test_dataloader)