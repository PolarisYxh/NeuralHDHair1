#不用深度图引入身体位姿，反过来 使用相机内外参矩阵（即把世界坐标变换到裁剪坐标）引入相机位姿
from solver.base_solver import BaseSolver
from Models.HairSpatNet import HairSpatNet
import torch.nn

from Tools.utils import *
import torch.autograd
from Models.Local_filter import Local_Filter
from torch.cuda.amp import autocast, GradScaler
class HairModelingHDCalSolver(BaseSolver):
    # use add_info to fine train
    @staticmethod
    def modify_options(parser):
        parser.set_defaults(save_root='checkpoints/HairModelingHDCal')
        parser.set_defaults(data_mode='imageCal')
        parser=HairSpatNet.modify_options(parser)
        parser=Local_Filter.modify_options(parser)
        parser.add_argument('--use_gan',action='store_true')
        parser.add_argument('--use_gt_Ori',action='store_true')


        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.opt = opt
        self.Spat_min_cha=opt.Spat_min_cha
        self.Spat_max_cha=opt.Spat_max_cha
        self.voxel2mesh_m = voxel2mesh_matrix(scale =2)
        self.initialize_networks(opt)

        if self.opt.isTrain:
            self.classification_weight=1.0
            self.classification_sparse_weight=0.1
            self.optimizer_global,self.optimizer_local=self.create_optimizers()
            self.criteria=torch.nn.CrossEntropyLoss()
            self.L1loss=torch.nn.L1Loss()
            self.L1loss_cont=torch.nn.L1Loss()


    def initialize_networks(self,opt):
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        self.net_global=HairSpatNet(opt,in_cha=opt.input_nc,min_cha=self.Spat_min_cha,max_cha=self.Spat_max_cha)#U-net
        self.net_local=Local_Filter(opt)

        self.net_global.print_network()
        self.net_local.print_network()
        print("save model to dir:"+os.path.join(opt.current_path,opt.save_root, opt.name,'checkpoint'))
        print("------------------------------------loading model")
        if not opt.no_use_pretrain:
            path=os.path.join(opt.current_path, opt.save_root, opt.pretrain_path)
            self.net_global=self.load_network(self.net_global,'HairSpatNet',opt.which_iter,opt,specify_path=path)
            print(path)
        else:
            self.net_global.init_weights(opt.init_type, opt.init_variance)
        if opt.continue_train or opt.isTrain is False:
            path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'checkpoint')
            print(f"---------------------------load model from {path}")
            if os.path.exists(path):
                print("---------------------------path exists")
                self.net_global = self.load_network(self.net_global, 'HairModelingGlobal', opt.which_iter, opt)
                self.net_local = self.load_network(self.net_local, 'HairModelingLocal', opt.which_iter, opt)
                print(os.path.join(path,'%s_%s.pth' % ('HairModelingGlobal', opt.which_iter)))
        else:
            print(" Training from Scratch! ")
            self.net_local.init_weights(opt.init_type, opt.init_variance)
        self.device = torch.device(f"cuda:{max(0,len(opt.gpu_ids)-2)}" if torch.cuda.is_available() else "cpu") 
        # self.net_global=nn.DataParallel(self.net_global, device_ids=opt.gpu_ids)
        self.net_global=self.net_global.to(self.device)
        # self.net_local=nn.DataParallel(self.net_local, device_ids=opt.gpu_ids)
        self.net_local=self.net_local.to(self.device)
        self.scaler = GradScaler()
        print(f"HairModelingHDCal正在使用的GPU编号是: {self.device.index}")



    def create_optimizers(self):
        params_global = []
        params_global += list(self.net_global.parameters())
        optimizer_global = torch.optim.Adam(params_global, lr=self.learning_rate, betas=(0.9, 0.999))
        params_local=list(self.net_local.parameters())
        optimizer_local = torch.optim.Adam(params_local, lr=self.learning_rate, betas=(0.9, 0.999))


        return optimizer_global,optimizer_local


    def preprocess_input(self,datas):
        """_summary_

        Args:
            datas (_type_): _description_

        Returns:
            calibration: 体素空间到裁剪空间的变换矩阵
        """        
        image = datas['image'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        gt_occ=datas['gt_occ']
        Ori2D = datas['Ori2D'].type(torch.float)
        add_info=datas['add_info'].type(torch.float)
        depth = datas['depth'].type(torch.float)
        calibration = datas['calibration'].type(torch.float)
        voxel2mesh_matrix = datas['voxel2mesh_matrix'].type(torch.float)
        calibration = calibration@voxel2mesh_matrix
        if self.use_gpu():
            image = image.to(self.device)
            gt_orientation = gt_orientation.to(self.device)
            gt_occ=gt_occ.to(self.device)
            Ori2D = Ori2D.to(self.device)
            add_info=add_info.to(self.device)
            depth=depth.to(self.device)
            calibration = calibration.to(self.device)
        return image,gt_orientation,gt_occ,Ori2D,add_info,depth,calibration



    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def train(self,iter_counter,dataloader,visualizer):
        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            if epoch > 20:
                self.opt.use_gt_Ori = False
            for i, datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()

                image,gt_orientation,gt_occ,Ori2D,add_info,depth,calibration= self.preprocess_input(datas)#ori.png(orientation map),(size,3,96,128,128),(size,1,96,128,128),
                # save_image(add_info,'test.png')
                # unsample = torch.nn.Upsample(scale_factor=self.opt.resolution[0]//96, mode='trilinear')
                # gt_occ_low=gt_occ.clone()
                # gt_orientation = unsample(gt_orientation)#size,3,96,128,128 to size,3,192,256,256
                # gt_occ=unsample(gt_occ)

                out_ori_hd, out_occ_hd, out_ori_low, out_occ_low, self.loss_local,self.loss_global= \
                self.net_local(image,add_info, gt_occ, gt_orientation, self.net_global, calibration=calibration, depth_map=depth,resolution=self.opt.resolution,mix=False)

                self.loss_backward(self.loss_global,self.optimizer_global,mix=False)
                # self.loss_backward(self.loss_global,self.optimizer_local)
                self.loss_backward(self.loss_local,self.optimizer_local,mix=False)


                visualizer.board_current_errors(self.loss_local)
                visualizer.board_current_errors(self.loss_global)
                if iter_counter.needs_printing():

                    losses = self.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                if iter_counter.needs_displaying():
                    # positive=torch.sigmoid(out_occ)>0.65
                    out_occ_hd[out_occ_hd>=0.2]=1
                    out_occ_hd[out_occ_hd<0.2]=0
                    out_occ_low[out_occ_low>=0.2]=1
                    out_occ_low[out_occ_low<0.2]=0
                    # out_occ=torch.where(positive,torch.ones_like(positive),torch.zeros_like(positive))
                    visualizer.draw_ori(add_info,gt_orientation,out_ori_hd*gt_occ,out_occ_hd*out_ori_hd,'_hd_L',mul=2)
                    visualizer.draw_ori(image,gt_orientation,out_ori_low*gt_occ,out_occ_low*out_ori_low,'_low_L',mul=1)

                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.net_local, 'HairModelingLocal', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.net_local, 'HairModelingLocal', 'latest', self.opt)
                    self.save_network(self.net_global, 'HairModelingGlobal', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.net_global, 'HairModelingGlobal', 'latest', self.opt)

                    iter_counter.record_current_iter()
            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()
            visualizer.print_epoch_errors(epoch, iter_counter.epoch_iter)
    def test(self,dataloader):
        with torch.no_grad():
            datas = dataloader.generate_test_data()
            image, gt_orientation, gt_occ = self.preprocess_input(datas)
            out_ori, out_occ = self.model(image)

            pred_ori=out_ori*gt_occ
            # pred_ori=(pred_ori+1)/2
            # gt_orientation=(gt_orientation+1)/2
            # save_image(pred_ori[:,:,45,:,:],'test1.png')
            # save_image(gt_orientation[:,:,45,:,:],'test2.png')
            pred_ori=pred_ori.permute(0,2,3,4,1)
            # pred_ori=torch.reshape(pred_ori,(128,128,96*3))
            pred_ori=pred_ori.cpu().numpy()
            save_ori_as_mat(pred_ori,self.opt)
    def inference(self,image,use_step,bust=None,calibration=None,norm_depth=None, use_bust=True,name=""):
        self.net_local.eval()
        self.net_global.eval()
        calibration = torch.from_numpy(calibration@self.voxel2mesh_m)#体素空间变换到裁剪空间
        with torch.no_grad():
            #以下相当于dataloader.generate_test_data()
            image=trans_image(image, self.opt.image_size)
            # if not use_step:
            #     image=trans_image(image, self.opt.image_size)#相当于get_image
            # else:
            #     parse = image[:, :, 2]
            #     image=image[:, :, [0, 1]]
            #     image=1-image
            #     image[np.where(parse<0.8)]=[0,0]
            image=torch.from_numpy(image)
            image=image.permute(2,0,1)
            Ori2D = image.clone()
            # image = get_Bust("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/DB1", image, self.opt.image_size)#TODO
            if isinstance(bust,np.ndarray) and use_bust:
                image = get_Bust2(bust,image,self.opt.image_size,name=name)
                # if name!="":
                #     return
            elif use_bust:
                image = get_Bust1(self.opt.current_path,image,self.opt.image_size)
            image = torch.unsqueeze(image, 0)
            Ori2D=torch.unsqueeze(Ori2D,0)
            # 以下相当于self.preprocess_input1
            image = image.type(torch.float)
            Ori2D = Ori2D.type(torch.float)
            # save_image(image,"1.png")
            if not self.opt.use_ori_addinfo:
                norm_depth = torch.from_numpy(norm_depth).unsqueeze(0).unsqueeze(0).type(torch.float)
                norm_depth = norm_depth.to(self.device) if self.use_gpu() else norm_depth
            if self.use_gpu():
                image = image.to(self.device)
                Ori2D = Ori2D.to(self.device)
                
            # if image.shape[1]==2:
            #     save_image(torch.cat([image, torch.zeros(1, 1, 256, 256).to(self.device)], dim=1)[:, :3, ...],f"{name}.png")
            # else:
            #     save_image(image,f"{name}.png")
            #image:带bust,strand2D:depth,Ori2D:不带bust的方向图,net_global,resolution,step=100000
            out_ori, out_occ,_,_ = self.net_local.test(image,norm_depth,Ori2D,self.net_global,self.opt.resolution, calibration=calibration)
                
            out_occ[out_occ>=0.2]=1
            out_occ[out_occ<0.2]=0
            pred_ori=out_ori*out_occ
            pred_ori=pred_ori.permute(0,2,3,4,1)#output:[1, 96, 128, 128, 3]
            pred_ori=pred_ori.cpu().numpy()
            pred_ori = save_ori_as_mat(pred_ori,self.opt,save=False,suffix="_"+str(self.opt.which_iter)+'_1')#[1, 128, 128, 3, 96]
            # show(pred_ori,scale=1)
            # pred_ori = save_ori_as_mat(pred_ori,self.opt,save=False,suffix="_"+str(self.opt.which_iter)+'_1')
            # 以下为save_ori_as_mat所做的操作
            # pred_ori=pred_ori * np.array([1, -1, -1])
            # pred_ori=pred_ori.transpose(0,2,3,4,1)
            # _,H,W,C,D=pred_ori.shape[:]
            # pred_ori=pred_ori.reshape(H ,W,C*D)
            return pred_ori,out_occ

    def loss_backward(self, losses, optimizer,retain=False,mix=False):
        optimizer.zero_grad()
        loss = sum(losses.values()).mean()
        if mix:
            self.scaler.scale(loss).backward(retain_graph=retain)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=retain)
            optimizer.step()

    def init_losses(self):
        self.total_loss = {}
        self.loss_global={}
        self.loss_local={}
    def get_latest_losses(self):
        self.total_loss={**self.loss_global,**self.loss_local}
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch % 30 == 0 and epoch != 0:
            self.learning_rate = self.learning_rate / 2

        for param_group_local in self.optimizer_local.param_groups:
            param_group_local['lr'] = self.learning_rate

        for param_group_global in self.optimizer_global.param_groups:
            param_group_global['lr'] = self.learning_rate
