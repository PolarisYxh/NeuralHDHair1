from dataload import data_loader
from Tools.utils import *
from solver.base_solver import BaseSolver
from Models.GrowingNet import GrowingNet
import os
import torch
import time
import numpy as np
import torch.nn
from Tools.resample import resample,process_list
class GrowingNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""
        parser.set_defaults(save_root='checkpoints/GrowingNet')
        parser.set_defaults(data_mode='strand')
        parser=GrowingNet.modify_options(parser)

        return parser

    def initialize(self,  opt):
        super().initialize( opt)


        self.opt = opt
        self.local_size = opt.local_size
        self.stride = opt.stride

        self.pt_num = opt.pt_per_strand
        self.sd_num = opt.sd_per_batch
        self.initialize_networks(opt)
        # input = torch.rand((3757, 3, 16, 16, 16)).cuda()
        # for i in range(10):
        #     _ = self.model_on_one_gpu.OriEncoder(input)
        # pos = torch.rand((1, 3, 12 * 12, 1)).cuda()
        # feat = torch.rand((1, 128, 12 * 12, 1)).cuda()
        # input = torch.cat([feat, pos], dim=1)
        # # input=feat
        # if opt.warm_gpu:
        #     for i in range(10):
        #         start=time.time()
        #         _ = self.model_on_one_gpu.Decoder_pos(input, pos)
        #         _ = self.model_on_one_gpu.Decoder_pos_Inv(input, pos)
        #         _ = self.model_on_one_gpu.Decoder_label(input, pos)
        #         print('test:',time.time()-start)
        #     torch.cuda.synchronize()

        if self.opt.isTrain:
            self.optimizer_GN=self.create_optimizers(opt)
            self.criteria=torch.nn.CrossEntropyLoss()
            self.L1loss=torch.nn.L1Loss()
        else:
            # self.roots = load_root(os.path.join(self.opt.current_path,"data/map_roots1024.data"))
            self.roots = scipy.io.loadmat(os.path.join(self.opt.current_path,"roots3.mat"), \
                        verify_compressed_data_integrity=False)['roots']
            # self.roots = self.roots[np.random.randint(0,self.roots.shape[0]-1,size=self.opt.num_root)]
            self.roots=transform(self.roots,scale=self.height//128)#[:,[2,1,0]]

    def create_optimizers(self,opt):
        GrowingNet_params=[]
        GrowingNet_params+=list(self.GrowingNet.parameters())
        optimizer_GN=torch.optim.Adam(GrowingNet_params,lr=self.learning_rate,betas=(0.9,0.999))

        return optimizer_GN

    def preprocess_input(self,datas):
        strands = datas['strands'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        labels = datas['labels']
        if labels is not None:
            labels=labels.type(torch.float)


        if len(self.opt.gpu_ids)>0:
            strands = strands.cuda()
            gt_orientation = gt_orientation.cuda()
            if labels is not None:
                labels = labels.cuda()
                labels = labels.permute(0, 3, 1, 2)#1, 1, 800, 72
        gt_orientation = gt_orientation.permute(0, 4, 1, 2, 3)#1,3,96,128,128
        strands = strands.permute(0, 3, 1, 2)#train:1,3,800,72; test:[1, 3, 15000, 1]

        return strands,gt_orientation,labels





    def initialize_networks(self, opt):
        self.GrowingNet= GrowingNet(opt,[self.depth, self.height, self.width], self.local_size, self.stride)
        self.GrowingNet.print_network()
        if len(opt.gpu_ids)>0:
            assert (torch.cuda.is_available())
            # self.GrowingNet.cuda()
            # self.model=torch.nn.DataParallel(self.GrowingNet,self.opt.gpu_ids)
            self.model=self.GrowingNet.cuda()
            # self.model_on_one_gpu=self.model.module.cuda()
            # self.GrowingNet=self.GrowingNet.module
        else:self.model=self.GrowingNet
        if opt.continue_train or opt.isTrain is False:
            path= os.path.join(opt.current_path,opt.save_root, opt.check_name,'checkpoint')
            if os.path.exists(path):
                self.GrowingNet=self.load_network(self.GrowingNet,'GrowingNet',opt.which_iter,opt)
        else:
            print(" Training from Scratch! ")
            self.GrowingNet.init_weights(opt.init_type,opt.init_variance)

    def train(self,iter_counter,dataloader,visualizer):

        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            for i,datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()
                # iter_counter.record_one_iteration()
                strands,gt_orientation,labels=self.preprocess_input(datas)

                in_points = strands[...,+1:-1]#1,3,800,72 to 1,3,800,70




                sta_points = strands[..., (2 * self.pt_num // 3) // 2:(2 * self.pt_num // 3) // 2 + 1]#取所有发丝的内部24索引点，用于测试并显示结果
                pre_points = strands[..., :-2]
                aft_points = strands[..., +2:]
                # print(pre_points[0,:,0,:])
                # print(aft_points[0,:,0,:])
                pre_labels = labels[..., +1:-1]
                # sum=torch.sum(pre_labels,dim=-1,keepdim=True).type(torch.long)
                # in_points[sum]=in_points[sum-1]
                # in_points[sum+1]=in_points[sum-1]
                # print(in_points.size())
                weight_labels=labels[...,+2:]
                weight_labels_Inv=labels[...,+1:-1]


                # if self.opt.Bidirectional_growth:
                #     out_points_1, labels_1, out_points_1_Inv, labels_1_Inv = self.model(in_points,gt_orientation,'nn')
                # else:
                #     out_points_1, labels_1 = self.model(in_points,gt_orientation,'nn')
                k=1
                if epoch>40 and self.opt.condition is not False:
                    k=3
                if epoch>100 and self.opt.condition is not False:
                    k=5
                if epoch>200 and self.opt.condition is not False:
                    k=10
                if k>=2:
                    self.total_loss['rnn_p_loss']=0
                    self.total_loss['rnn_p_loss_Inv']=0
                if self.opt.Bidirectional_growth:
                    all_out_points,_,all_out_points_inv_,_=self.model(in_points,gt_orientation,k,'rnn')
                self.total_loss['p_loss']=self.L1loss(aft_points*weight_labels,all_out_points[...,0:self.pt_num-2]*weight_labels)
                self.total_loss['p_loss_Inv']=self.L1loss(pre_points*weight_labels_Inv,all_out_points_inv_[...,0:self.pt_num-2]*weight_labels_Inv)
                for i in range(1,k):
                    self.total_loss['rnn_p_loss']+=self.L1loss(aft_points[...,i:]*weight_labels[...,i:],all_out_points[...,i*(self.pt_num-2):(i+1)*(self.pt_num-2)-i]*weight_labels[...,i:])
                    self.total_loss['rnn_p_loss_Inv']+=self.L1loss(pre_points[...,:-i]*weight_labels_Inv[...,i:],all_out_points_inv_[...,i*(self.pt_num-2)+i:(i+1)*(self.pt_num-2)]*weight_labels_Inv[...,i:])

                # print('pred',out_points_1[0,:,0,0:5])
                # print('gt', aft_points[0,:, 0, 0: 5])
                # print('back:',out_points_1_Inv[0,:,0,:5])
                # print('gt:',pre_points[0,:,0,:5])


                # print('gt:',aft_points[0,:,0,0:3])
                # print('out:',out_points_1[0,:,0,0:3])
                drap_gt_points = in_points * pre_labels
                pre_labels = pre_labels.type(torch.long)[:,0,...]


                ####losss
                # self.total_loss['p_loss']=torch.mean(torch.abs(aft_points-out_points_1)*weight_labels)
                # self.total_loss['p_loss']=self.L1loss(aft_points*weight_labels,out_points_1*weight_labels)

                # if self.opt.pred_label:
                #     self.total_loss['classify_loss'] = torch.mean(self.criteria(labels_1,pre_labels)) * 0.1
                # if self.opt.Bidirectional_growth:
                #     self.total_loss['p_loss_Inv']=torch.mean(torch.abs(pre_points-out_points_1_Inv)*weight_labels_Inv)
                    # self.total_loss['p_loss_Inv']=self.L1loss(pre_points*weight_labels_Inv,out_points_1_Inv*weight_labels_Inv)

                #####backward
                self.loss_backward(self.total_loss,self.optimizer_GN)
                visualizer.board_current_errors(self.total_loss)
                if iter_counter.needs_printing():
                    losses = self.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter,losses, iter_counter.time_per_iter)
                    # print(labels_1[0,:,0,:])

                if iter_counter.needs_displaying():
                    with torch.no_grad():
                        if self.opt.Bidirectional_growth:
                            out_points_2, labels_2, out_points_2_Inv, labels_2_Inv =self.model(sta_points,gt_orientation,self.pt_num//2,'rnn')
                        else:
                            out_points_2, labels_2 = self.model(sta_points,gt_orientation,self.pt_num//2,'rnn')

                    if self.opt.pred_label:
                        drap_points = out_points_2 * torch.where(labels_2[:, 1:2, ...] > labels_2[:, 0:1, ...],torch.ones_like(labels_2[:, 0:1, ...]),torch.zeros_like(labels_2[:, 0:1, ...]))
                        if self.opt.Bidirectional_growth:
                            drap_points_Inv = out_points_2_Inv * torch.where(labels_2_Inv[:, 1:2] > labels_2_Inv[:, 0:1],torch.ones_like(labels_2_Inv[:, 0:1]),torch.zeros_like(labels_2_Inv[:, 0:1]))
                    else:
                        drap_points = out_points_2
                        if self.opt.Bidirectional_growth:
                            drap_points_Inv = out_points_2_Inv

                    N = out_points_2.size(1)
                    s = np.random.randint(0, N)
                    drap_gt_points=drap_gt_points.permute(0,2,3,1)
                    drap_points=drap_points.permute(0,2,3,1)
                    drap_points=drap_points.cpu().detach().numpy()
                    drap_gt_points=drap_gt_points.cpu().detach().numpy()
                    # visualizer.draw_3d(drap_gt_points[0], drap_points[0], s,self.width,self.height,self.depth, "Forward3d")
                    # visualizer.draw_samples(drap_gt_points[0], drap_points[0], s,self.width,self.height,self.depth, "Forward")
                    if self.opt.Bidirectional_growth:
                        drap_points_Inv = drap_points_Inv.permute(0, 2, 3, 1)
                        drap_points_Inv = drap_points_Inv.cpu().detach().numpy()
                        # visualizer.draw_3d(drap_gt_points[0], drap_points_Inv[0], s,self.width,self.height,self.depth, "Backward3d")
                        # visualizer.draw_samples(drap_gt_points[0], drap_points_Inv[0], s,self.width,self.height,self.depth, "Backward")


                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.GrowingNet, 'GrowingNet', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.GrowingNet, 'GrowingNet', 'latest', self.opt)
                    iter_counter.record_current_iter()
            visualizer.print_epoch_errors(epoch, iter_counter.epoch_iter)
            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()

    def get_pred_strands(self,datas,ori_orient=None,use_rule=False):
        strands, gt_orientation,labels = self.preprocess_input(datas)
        if use_rule:
            num_strand = strands.shape[2]
            hair_strands = torch.zeros(100, 3, num_strand).cuda()
            curr_node = strands.squeeze()
            hair_strands[0] = curr_node#gt_orientation按照前后，上下，左右顺序;hair_strands,strands相反按照左右，上下，前后;curr_node_orien按照左右，上下，前后
            for i in range(1,99):
                index = curr_node.to(torch.long)
                index[0:2]=torch.clip(index[0:2],0,gt_orientation.shape[3]-1)
                index[2:]=torch.clip(index[2:],0,gt_orientation.shape[2]-1)
                curr_node_orien = gt_orientation[0,:,index[2],index[1],index[0]]#1,3,96,128,128
                # curr_node_orien = curr_node_orien*x
                # curr_node_orien = torch.flip(curr_node_orien,dims=[0])#curr_node_orien变为按照前后，上下，左右
                hair_strands[i] = hair_strands[i-1] + 0.8 * curr_node_orien
                curr_node = hair_strands[i]
            hair_strands[:,:2,:] = torch.clip(hair_strands[:,:2,:],0,gt_orientation.shape[3]-1)
            hair_strands[:,2,:] = torch.clip(hair_strands[:,2,:],0,gt_orientation.shape[2]-1)
            out_points_2 = hair_strands.permute(1,2,0).unsqueeze(0)
            
            hair_strands_inv = torch.zeros(100, 3, num_strand).cuda()
            curr_node = strands.squeeze()
            hair_strands_inv[0] = curr_node
            for i in range(1,99):
                index = curr_node.to(torch.long)
                index[0:2]=torch.clip(index[0:2],0,gt_orientation.shape[3]-1)
                index[2:]=torch.clip(index[2:],0,gt_orientation.shape[2]-1)
                curr_node_orien = gt_orientation[0,:,index[2],index[1],index[0]]

                hair_strands_inv[i] = hair_strands_inv[i-1] - 0.8*curr_node_orien
                curr_node = hair_strands_inv[i]
            hair_strands_inv[:,:2,:] = torch.clip(hair_strands_inv[:,:2,:],0,gt_orientation.shape[3]-1)
            hair_strands_inv[:,2,:] = torch.clip(hair_strands_inv[:,2,:],0,gt_orientation.shape[2]-1)
            out_points_2_Inv = hair_strands_inv.permute(1,2,0).unsqueeze(0)
            pt_num = 100
        else:
            with torch.no_grad():
                if self.opt.Bidirectional_growth:
                    pt_num = self.pt_num//2 #default is self.pt_num-1
                    print('begin.....')
                    start = time.time()
                    # wcenters,wlatents=self.model_on_one_gpu.encoder(gt_orientation)
                    # wlatents=wlatents.expand(len(self.opt.gpu_ids),*wlatents.size()[1:])
                    # print(wlatents.size())
                    # wcenters=wcenters.expand(len(self.opt.gpu_ids),*wcenters.size()[1:])
                    # print('encoder cost:',time.time()-start)
                    # gt_orientation = close_voxel(gt_orientation, 5)
                    gt_orientation=gt_orientation.expand(len(self.opt.gpu_ids),*gt_orientation.size()[1:])
                    out_points_2, labels_2, out_points_2_Inv, labels_2_Inv=self.model(strands,gt_orientation,pt_num,'rnn')
                    # torch.cuda.synchronize()
                    print('grow cost:', time.time() - start)

                else:
                    pt_num = self.pt_num #default is self.pt_num-1
                    out_points_2, labels_2 = self.model(strands,gt_orientation,pt_num,'rnn')


        print(out_points_2.size())
        gt_orientation=gt_orientation.permute(0,2,3,4,1)
        out_points_2=out_points_2.permute(0,2,3,1)
        if self.opt.pred_label:
            labels_2=labels_2.permute(0,2,3,1)

        out_points_2 = out_points_2.reshape(1, -1,  pt_num, 3)
        if self.opt.pred_label:
            labels_2 = labels_2.reshape(1, -1,  pt_num,2)

        gt_orientation=gt_orientation.cpu().numpy()
        out_points_2=out_points_2.cpu().numpy()
        if self.opt.pred_label:
            labels_2=labels_2.cpu().numpy()
        if self.opt.Bidirectional_growth:

            out_points_2_Inv = out_points_2_Inv.permute(0, 2, 3, 1)
            if self.opt.pred_label:
                labels_2_Inv = labels_2_Inv.permute(0, 2, 3, 1)
                labels_2_Inv=labels_2_Inv.reshape(1,-1,pt_num,2)
                labels_2_Inv=labels_2_Inv.cpu().numpy()
            out_points_2_Inv=out_points_2_Inv.reshape(1,-1,pt_num,3)
            out_points_2_Inv=out_points_2_Inv.cpu().numpy()
        if isinstance(ori_orient,np.ndarray):
            mask = np.linalg.norm(ori_orient[None,...], axis=-1)
        else:
            mask = np.linalg.norm(gt_orientation, axis=-1)

        strand_delete_by_ori, segment = delete_point_out_ori(mask, out_points_2)
        if self.opt.pred_label:
            strand_delete_by_label, segment_label = delete_point_out_label(out_points_2, labels_2)
        if self.opt.Bidirectional_growth:
            strand_delete_by_ori_Inv, segment_Inv = delete_point_out_ori(mask, out_points_2_Inv)
            if self.opt.pred_label:
                strand_delete_by_label_Inv, segment_label_Inv = delete_point_out_label(out_points_2_Inv, labels_2_Inv)
        else:
            segment_Inv = None
            segment_label_Inv = None
            strand_delete_by_ori_Inv = None
            strand_delete_by_label_Inv = None

        final_strand_del_by_ori, final_segment = concat_strands(strand_delete_by_ori, strand_delete_by_ori_Inv, segment,
                                                                segment_Inv, self.opt.Bidirectional_growth,gt_orientation.shape[2]//96)
        if self.opt.pred_label:
            final_strand_del_by_label, final_segment_label = concat_strands(strand_delete_by_label,
                                                                            strand_delete_by_label_Inv, segment_label,
                                                                            segment_label_Inv,
                                                                            self.opt.Bidirectional_growth)
        return final_strand_del_by_ori,final_segment
    def test(self,dataloader):
        if self.opt.Bidirectional_growth:
            datas=dataloader.generate_random_root()
        else:
            datas=dataloader.generate_test_data(self.opt.growInv)
        final_strand_del_by_ori,final_segment=self.get_pred_strands(datas)
        write_strand(final_strand_del_by_ori, self.opt, final_segment, 'ori')
        # write_strand(final_strand_del_by_label, self.opt, final_segment_label, 'label')
    def generate_random_root_from_roots(self):
        roots1 = self.roots.astype('int')
        occ=np.linalg.norm(self.gt_orientation,axis=-1)[0]
        # occ=(occ>0).astype(np.float32)
        occ1 = occ[roots1[:,2],roots1[:,1],roots1[:,0]]
        sample_index = np.where(occ1>0)
        # random_points = self.roots[sample_index]
        # random_points = random_points[np.random.randint(0,random_points.shape[0]-1,size=self.opt.num_root)]
        
        samle_voxel_index= np.unique(roots1[sample_index], axis=0)
        random_points=samle_voxel_index[np.random.randint(0,samle_voxel_index.shape[0]-1,size=self.opt.num_root)]
        random_points=random_points+np.random.random(random_points.shape[:])[None]
        
        # random_points = self.roots
        random_points=random_points[...,None,:]
        self.gt_orientation=torch.from_numpy(self.gt_orientation)
        random_points=torch.from_numpy(random_points)
        random_points=torch.reshape(random_points,(len(self.opt.gpu_ids),-1,1,3))
        return_list={
            'gt_ori': self.gt_orientation,
            'strands': random_points,
            'labels': None
        }

        return return_list
    def generate_random_root(self,occ):
        samle_voxel_index =np.where(occ>0)
        samle_voxel_index=np.array(samle_voxel_index)
        samle_voxel_index=samle_voxel_index.transpose(1,0)
        
        back = np.min(samle_voxel_index[:,0], axis=0)#前后
        front = np.max(samle_voxel_index[:,0], axis=0)
        low1 = np.min(samle_voxel_index[:,1], axis=0)#上下
        high1 = np.max(samle_voxel_index[:,1], axis=0)
        left = np.min(samle_voxel_index[:,2], axis=0)#左右
        right = np.max(samle_voxel_index[:,2], axis=0)
        mid = (low1+high1)//2
        self.pt_num = int((high1-low1)*3)
        low = np.min(self.roots[:,1],axis=0)
        high = np.max(self.roots[:,1],axis=0)
        samle_voxel_index2 = samle_voxel_index[np.where(samle_voxel_index[:,1]<=high)][...,:3]#对头皮毛孔以上的体素进行采样
        # if high1-low1>90:
        #     random_points=samle_voxel_index[np.random.randint(0,samle_voxel_index.shape[0]-1,size=self.opt.num_root*2)]
        if mid>high:
            samle_voxel_index1 = samle_voxel_index[np.where(samle_voxel_index[:,1]==mid)][...,:3]
            random_points=samle_voxel_index1[np.random.randint(0,samle_voxel_index1.shape[0]-1,size=self.opt.num_root)]
            random_points=np.append(random_points,samle_voxel_index2[np.random.randint(0,samle_voxel_index2.shape[0]-1,size=self.opt.num_root//3)],axis=0)
            random_points=samle_voxel_index2[np.random.randint(0,samle_voxel_index2.shape[0]-1,size=self.opt.num_root//3)]
        else:
            self.pt_num = 72
            random_points=samle_voxel_index2[np.random.randint(0,samle_voxel_index2.shape[0]-1,size=self.opt.num_root)]
        random_points=random_points[:,::-1]+np.random.random(random_points.shape[:])[None]
        random_points=random_points[...,None,:]
        
        self.gt_orientation=torch.from_numpy(self.gt_orientation)
        random_points=torch.from_numpy(random_points)
        random_points=torch.reshape(random_points,(len(self.opt.gpu_ids),-1,1,3))

        return_list={
            'gt_ori': self.gt_orientation,
            'strands': random_points,
            'labels': None
        }

        return return_list
    def generate_test_data(self,growInv=False):
        self.segments=np.array(self.segments)
        index=np.cumsum(self.segments)
        if growInv:
            index=index-1
        else:
            index=index[:-1]

        strands=self.points[index][None]

        self.gt_orientation = torch.from_numpy(self.gt_orientation)
        strands=strands[...,None,:]
        strands=torch.from_numpy(strands)
        return_list={
            'gt_ori': self.gt_orientation,
            'strands':strands,
            'labels': None
        }
        return return_list
    def index(self,feat,uv):
        '''
        :param feat: [B, C, H, W] image features
        :param uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]
        :return: [B, C, N] sampled pixel values
        '''
        uv=uv.unsqueeze(0).unsqueeze(0).to(torch.double)
        feat=feat.unsqueeze(0).permute((0,3,1,2)).to(torch.double)
        samples=torch.nn.functional.grid_sample(feat, uv, mode='bilinear',align_corners=True)
        return samples[0,:,0,:]
    def inference(self, ori, matrix,hair_img, avg_color,sample_num=-1):
        self.model.eval()
        with torch.no_grad():
            # ori = scipy.io.loadmat("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/DB1/Ori_gt.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
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

            new_gt_occ = np.dot(gt_occ1, matrix)+np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])
            new_gt_occ = new_gt_occ.T.astype('int')
            
            index = (new_gt_occ[2] >= 0) & (new_gt_occ[2] <= s[2]-1)&(new_gt_occ[0] >= 0) & (new_gt_occ[0] <= s[0]-1)&(new_gt_occ[1] >= 0) & (new_gt_occ[1] <= s[1]-1)
            new_gt_occ = new_gt_occ[:,index]
            mask1 = mask1[:,index]
            ori1 = ori[tuple(mask1)]
            new_ori1 = np.dot(ori1.reshape((-1,3)),matrix)
            ori = np.zeros_like(ori)
            ori[new_gt_occ[0],new_gt_occ[1],new_gt_occ[2]] = new_ori1
            
            ori = ori[::-1, :, :,  :]
            ori = np.transpose(ori, (1,0,2,3)) 
            # 转换+填充voxel
            ori = ori.transpose(2, 0, 1, 3)# 转换后ori: 96*128*128*3
            occ=np.linalg.norm(ori,axis=-1)
            occ=(occ>0).astype(np.float32)[...,None]
            occ1 = torch.from_numpy(occ).permute((3,0,1,2))
            k=3
            p=int(k/2)
            ori,dilate_ori,occ2,dilate_occ=close_voxel1(occ1,torch.from_numpy(ori.copy()).permute((3,0,1,2)),k)
            # 方向场周围包一圈指向方向场的方向
            # Define 3D Sobel operator
            sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).unsqueeze(0).float()*(-1)#左右

            sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                    [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).unsqueeze(0).float()#上下

            sobel_z = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0 ,0]],
                                    [[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]]]).unsqueeze(0).float()#前后

            occ3=occ2.unsqueeze(0)
            # Apply Sobel operator
            G_x = F.conv3d(occ3, sobel_x, stride=1,padding='same').squeeze(0)
            G_y = F.conv3d(occ3, sobel_y, stride=1,padding='same').squeeze(0)
            G_z = F.conv3d(occ3, sobel_z, stride=1,padding='same').squeeze(0)

            # Calculate magnitude
            G = torch.sqrt(G_x**2 + G_y**2 + G_z**2)
            ori_edge = torch.concat((G_x,G_y,G_z),0)/G
            ori_edge = torch.where(torch.isnan(ori_edge), 0,ori_edge)
            ori_edge = torch.where(occ2>0,ori, ori_edge)
            
            # ori_edge1=F.avg_pool3d(ori_edge,kernel_size=3, stride=1, padding=1)
            # mask = torch.norm(ori_edge, dim=0)
            # ori_edge = torch.where(mask>0, ori_edge, ori_edge1)
            # show_slice(ori_edge.permute((2,3,0,1)).numpy(),img = np.zeros((1024,1024,3)),mode=2)
            # 腐蚀occ，作为采样的occ
            k=4
            p=k//2
            occ1 = 1-F.max_pool3d(1-dilate_occ, kernel_size=k, stride=1, padding=p)
            # draw_circles_by_projection(occ1,iter=3)
            occ = occ1.cpu().numpy().transpose(1,2,3,0)
            # 使用膨胀后的方向场进行生长，防止断发
            ori1 = ori_edge
            ori1=ori1.cpu().numpy().transpose(1, 2, 3, 0)
            if transfer:
                gt_orientation= ori1*np.array([1,-1,-1])  # scaled
            else:
                gt_orientation= ori1
            self.gt_orientation = gt_orientation[None]
            if self.opt.Bidirectional_growth:
                datas=self.generate_random_root(occ)
            else:
                datas=self.generate_random_root_from_roots()
                # datas=self.generate_test_data(self.opt.growInv)
            final_strand_del_by_ori,final_segment = self.get_pred_strands(datas,ori_orient=ori.cpu().numpy().transpose((1,2,3,0)),\
                                                                          use_rule=True)#ori_orient=ori.cpu().numpy().transpose((1,2,3,0)),
            # 采样点
            if sample_num!=-1:
                final_strand_del_by_ori = process_list(final_strand_del_by_ori,final_segment,sample_num)
                final_segment = (np.ones(len(final_strand_del_by_ori))*sample_num).astype("int")
                #删除大部分在膨胀区域的发丝
                # final_strand_del_by_ori,final_segment=delete_strand_out_ori(occ2,final_strand_del_by_ori,final_segment)#occ2:[1, 96, 128, 128]
                final_strand_del_by_ori = final_strand_del_by_ori.reshape(-1,3)
            # x=np.array(final_strand_del_by_ori)[:,:].astype('int')
            # draw_circles_by_projection1(x)
            final_strand_del_by_ori1=(np.array(final_strand_del_by_ori[:,[0,1,2]])-np.array([s[0]//2,s[1]//2,s[2]//2]))*np.array([-1,1,1])
            final_strand_del_by_ori1 = (np.dot(final_strand_del_by_ori1, np.linalg.inv((matrix)))*np.array([-1,1,1])+np.array([s[0]//2,s[1]//2,s[2]//2]))[:,[0,1,2]]
                                        
            # final_strand_del_by_ori1[:,0] = final_strand_del_by_ori1[:,0]-1.42 
            final_strand_del_by_ori2 = torch.from_numpy(np.array(final_strand_del_by_ori1)[:,[0,1]]/(s[0]//2))-1
            # 膨胀头发分割图,并进行采样
            mask = np.sum(hair_img,axis=2)
            mask[mask!=0]=1
            mask = torch.from_numpy(mask.astype('float32')[...,None])
            mask1=F.max_pool2d(mask, kernel_size=k*4+1, stride=1, padding=int(k*4/2))
            mask2 = mask1-mask
            hair_img[np.where(mask2==1)[:2]]=avg_color[:3]
            mask3 = 1-F.max_pool2d(1-mask, kernel_size=k*4+1, stride=1, padding=int(k*4/2))
            mask3 = mask-mask3
            hair_img = torch.from_numpy(hair_img/255)
            # save_image(hair_img.permute((2,0,1)),"./texture3.png")
            hair_img1=F.avg_pool2d(hair_img.permute((2,0,1)), kernel_size=k*4+1, stride=1, padding=int(k*4/2)).permute((1,2,0))
            hair_img=hair_img*(1-mask3[...,[0,0,0]])+hair_img1*mask3[...,[0,0,0]]
            # save_image(hair_img.permute((2,0,1)),"./texture4.png")
            colors = self.index(hair_img,final_strand_del_by_ori2).permute((1,0)).numpy()
            colors = np.c_[colors, np.ones(len(colors))]
            colors = (colors*255).astype('uint8')
            final_strand_del_by_ori = transform_Inv(final_strand_del_by_ori,scale=scale)
        # write_strand(final_strand_del_by_ori, self.opt, final_segment, 'ori')
        return final_strand_del_by_ori,final_segment,colors
        # write_strand(final_strand_del_by_label, self.opt, final_segment_label, 'label')



    def loss_backward(self, losses, optimizer):
        optimizer.zero_grad()
        loss=sum(losses.values()).mean()
        loss.backward()
        optimizer.step()

    def init_losses(self):
        self.total_loss={}

    def get_latest_losses(self):
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch % self.opt.lr_update_freq==0 and epoch!=0:
            self.learning_rate=self.learning_rate/2
        for params in self.optimizer_GN.param_groups:
            params['lr']=self.learning_rate















