from solver.base_solver import BaseSolver
from Models.HairSpatNet import HairSpatNet
from Models.UNet import U_Net
from Loss.loss import lovasz_hinge,uniform_sample_loss,probability_sample_loss,binary_cross_entropy,compute_gradient_penalty
from Loss.percepture_loss import VGGLoss
import os
import torch
import torch.nn
import torch.nn.functional as F
from Tools.utils import *
import torch.autograd
from Tools.drawTools import draw_arrows_by_projection2
class StepNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        parser.set_defaults(save_root='checkpoints/StepNet')
        parser.set_defaults(data_mode='step')
        # parser=HairSpatNet.modify_options(parser)
        # parser.add_argument('--close_gt',default=False)
        parser.add_argument('--use_gan',action='store_true')
        # parser.add_argument('--use_gt_Ori',action='store_true')
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.opt = opt
        # self.Spat_min_cha=opt.Spat_min_cha
        # self.Spat_max_cha=opt.Spat_max_cha

        self.initialize_networks(opt)

        if self.opt.isTrain:
            self.classification_weight=1.0
            self.classification_sparse_weight=0.1
            self.optimizer=self.create_optimizers()
            self.L1loss = torch.nn.L1Loss(reduction='sum')
            self.crit_vgg = VGGLoss(model='vgg19', gpu_ids=opt.gpu_ids, layer=35)
            # self.criteria=torch.nn.CrossEntropyLoss()
            # self.L1loss=torch.nn.L1Loss()
            # self.L1loss_cont=torch.nn.L1Loss()


    def initialize_networks(self,opt):
        # self.net=HairSpatNet(opt,in_cha=opt.input_nc,min_cha=self.Spat_min_cha,max_cha=self.Spat_max_cha)
        self.net=U_Net(img_ch=3,output_ch=3)
        self.net.print_network()
        if opt.continue_train or opt.isTrain is False:
            path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'checkpoint')
            if os.path.exists(path):
                self.net = self.load_network(self.net, 'StepNet', opt.which_iter, opt)
            else:
                print(path+" not exists!")
                exit()
        else:
            print(" Training from Scratch! ")
            self.net.init_weights(opt.init_type, opt.init_variance)
            
        from torchsummary import summary
        summary(self.net, input_size=(3, 256, 256), device='cpu')
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.model=self.net
            self.model=self.model.cuda()
    
    def create_optimizers(self):
        params = []
        params += list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999))

        return optimizer


    def preprocess_input(self,datas):
        input = datas["input"].type(torch.float)
        gt_feat = datas["gt_feat"].type(torch.float)
        gt_sum=datas["gt_sum"]
        target = datas["target"].type(torch.float)

        if self.use_gpu():
            input = input.cuda()
            gt_feat = gt_feat.cuda()
            gt_sum=gt_sum.cuda()
            target = target.cuda()
        # save_image(torch.cat([image,torch.zeros(1,1,256,256).cuda()],dim=1),'test_image.png')
        # save_image(depth,'test_depth.png')
        return input,gt_feat,gt_sum,target
    def preprocess_input1(self,datas):
        image = datas['image'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        gt_occ=datas['gt_occ']
        Ori2D = datas['Ori2D'].type(torch.float)
        
        if self.use_gpu():
            image = image.cuda()
            gt_orientation = gt_orientation.cuda()
            gt_occ=gt_occ.cuda()
            Ori2D = Ori2D.cuda()
        # save_image(torch.cat([image,torch.zeros(1,1,256,256).cuda()],dim=1),'test_image.png')
        # save_image(depth,'test_depth.png')
        return image,gt_orientation,gt_occ,Ori2D


    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def train(self,iter_counter,dataloader,test_dataloader,visualizer):#单张图loss最小0.040
        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            self.model.train()
            for i, datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()
                input,gt_feat,gt_sum,target = self.preprocess_input(datas)
                
                out_img = self.model(input)

                self.G_loss["train_loss"] = 0.1*self.crit_vgg(out_img, gt_feat, target_is_features=True)
                self.G_loss["train_loss"] += self.L1loss(out_img, target)/(3*gt_sum.squeeze().sum())
                
                self.loss_backward(self.G_loss, self.optimizer,False)


                losses = self.get_latest_losses()
                visualizer.board_current_errors(losses)
                if iter_counter.needs_printing():

                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                if iter_counter.needs_displaying():#every 20 steps
                    save_image(out_img[0],"step_display.png")

                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.model, 'StepNet', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.model, 'StepNet', 'latest', self.opt)

                    iter_counter.record_current_iter()
            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()
            self.model.eval()
            with torch.no_grad():
                for i, datas in enumerate(test_dataloader):
                    self.init_losses()
                    input,gt_feat,gt_sum,target = self.preprocess_input(datas)
                    input,gt_feat,gt_sum,target = input[None],gt_feat[None],gt_sum[None],target[None]
                    out_img = self.model(input)
                    
                    self.G_loss["test_loss"] = 0.1*self.crit_vgg(out_img, gt_feat, target_is_features=True)
                    self.G_loss["test_loss"] += self.L1loss(out_img, target)/(3*gt_sum.squeeze().sum())
                    visualizer.board_current_errors(self.G_loss)
                    if i==0:
                        save_image(out_img[0],"test_step_display.png")
                    
            visualizer.print_epoch_errors(epoch, iter_counter.epoch_iter)
            # test_loss /= len(test_dataloader.dataset)
            # accuracy = 100.0 * correct / len(test_loader.dataset)
            # print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
            

    def test(self,dataloader):
        with torch.no_grad():
            datas = dataloader.generate_test_data()
            image, gt_orientation, gt_occ, ori2d = self.preprocess_input1(datas)
            out_ori, out_occ = self.model.test(image,ori2d)
            pred_ori=out_ori*gt_occ
            pred_ori=pred_ori.permute(0,2,3,4,1)
            pred_ori=pred_ori.cpu().numpy()
            ori = save_ori_as_mat(pred_ori,self.opt,suffix="_"+str(self.opt.which_iter))
            
            out_occ[out_occ>=0.2]=1
            out_occ[out_occ<0.2]=0
            pred_ori=out_ori*out_occ
            pred_ori=pred_ori.permute(0,2,3,4,1)
            pred_ori=pred_ori.cpu().numpy()
            ori = save_ori_as_mat(pred_ori,self.opt,suffix="_"+str(self.opt.which_iter)+'_1')
            
    def inference(self,image):
        self.model.eval()
        with torch.no_grad():
            image= cv2.resize(image,(256,256)).transpose([2,0,1])
            image=torch.from_numpy(image) / 255
            image = image.type(torch.float)
            if self.use_gpu():
                image = image.cuda()
            
            image = image[None]
            out_img = self.model(image)
            # img = out_img[0][[2, 1, 0], :, :,]#不可导，使用矩阵乘法进行通道重排才可导
            # # img = img.cpu().numpy().transpose([1,2,0])
            # # cv2.imwrite("test_step_display.png",img)
            # save_image(img,"test_step_display.png")
            return out_img[0].permute(1, 2, 0).to("cpu").numpy()
    def loss_backward(self, losses, optimizer,retain=False):
        optimizer.zero_grad()
        loss = sum(losses.values()).mean()
        loss.backward(retain_graph=retain)
        optimizer.step()

    def init_losses(self):
        self.total_loss = {}
        # self.D_loss={}
        self.G_loss={}

    def get_latest_losses(self):
        self.total_loss={**self.G_loss}
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch % self.opt.lr_update_freq == 0 and epoch != 0:
            self.learning_rate = self.learning_rate / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        # for param_group_d in self.D_optimizer.param_groups:
        #     param_group_d['lr'] = self.learning_rate
        print(f"update lr to :{self.learning_rate}")
