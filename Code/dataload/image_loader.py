from dataload.base_loader import base_loader
from Tools.utils import *
import torch
import shutil
from dataload.render_strand import render_strand
import trimesh
from skimage import transform as trans
import cv2
class image_loader(base_loader):

    def initialize(self,opt):

        self.opt = opt
        self.batch_size = opt.batch_size
        self.image_size = opt.image_size
        self.isTrain = opt.isTrain
        self.parent_dir = opt.current_path
        self.root = os.path.join(self.parent_dir,opt.strand_dir)
        self.mesh = trimesh.load(os.path.join(os.path.dirname(__file__),"../../",'female_halfbody_medium.obj'))
        self.orig_vertices = self.mesh.vertices.copy()
        self.orig_vertices = self.orig_vertices+np.array([0.00703544,-1.58652416,-0.01121912])
        if self.isTrain:
            self.num_of_val = opt.num_of_val
            self.train_corpus = []
            self.train_nums = 0
            self.generate_corpus()
        else:
            self.train_corpus = []
            self.train_nums = 0
            self.generate_corpus(is_train=False)
    def get_test_data(self,dirs,is_rot=False):
        data=[]
        files=os.listdir(dirs)
        files=sorted(files)
        #Delete data with number greater than 600
        # for file in files[:50]:
        #     # if is_rot==False:
        #     if "DB1" in file:
        #         # continue
        data.append(os.path.join(dirs,"DB1"))
        return data
    def generate_corpus(self,is_train=True):
        # exclude the tail, to do improve this
        if is_train:
            self.all_data = get_all_the_data1(self.root,self.opt.is_rot)
            self.train_corpus=self.all_data
        else:
            self.train_corpus = self.get_test_data(self.root,self.opt.is_rot)

        self.train_nums = len(self.train_corpus)
        if is_train:
            print(f"num of training data: {self.train_nums}")
        else:
            print(f"num of test data: {self.train_nums}")
    def __len__(self):
        # if self.isTrain:
        #     numbers = list(range(0, 13))
        #     # 随机选择三个不重复的数字
        #     random_numbers = list(random.sample(numbers, 3))
        #     self.train_corpus=np.array(self.all_data).reshape((-1,13))
        #     self.train_corpus=self.train_corpus[:,random_numbers].reshape((-1))
        #     print("train_data random:"+str(len(self.train_corpus)))
        random.shuffle(self.train_corpus)
        return len(self.train_corpus)
    def __getitem__(self, index):
        data_list={}
        file_name=self.train_corpus[index]
        x=[False,True]
        flip=x[random.randint(0, 1)]
        strand1 = np.load(os.path.join(file_name,os.path.basename(file_name)+".npy"))
        strand1  = strand1.reshape((-1,3))
        
        strand = strand1.copy()
        strand = strand+np.array([0.00703544,-1.58652416,-0.01121912])
        
        x=random.randint(-30,30)
        # x=-x if random.randint(0,1)==1 else x
        y=random.randint(-20,20)
        # y=-y if random.randint(0,1)==1 else y
        ang = [y,x]
        tform = trans.SimilarityTransform(rotation=[np.deg2rad(ang[0]),np.deg2rad(ang[1]),np.deg2rad(0)],dimensionality=3)#[0,30,0] 从上往下看顺时针旋转v3；[15,0,0] 向下旋转v1
        strand = trans.matrix_transform(strand, tform.params)+np.array([-0.00703544,1.58652416,0.01121912])
        self.mesh.vertices = trans.matrix_transform(self.orig_vertices, tform.params)+np.array([-0.00703544,1.58652416,0.01121912])
        
        strand1 = strand.reshape((-1,100,3))
        strand_before = strand1[:,:-1,:]
        strand_aft = strand1[:,1:,:]
        ori1 = strand_aft-strand_before
        ori2 = strand1[:,99,:]-strand1[:,98,:]
        ori_list = np.append(ori1,ori2[:,None],axis=1)
        ori_list[:,:,2]= 0
        norms = np.linalg.norm(ori_list, axis=2, keepdims=True)
        # Handle elements with norm of zero separately
        zero_norm_indices = np.isclose(norms, 0.0)
        norms[zero_norm_indices] = 1.0
        ori_list = ori_list/norms
        ori_list = ori_list.reshape((-1,3))
        ori_list[:,0]= ori_list[:,0]/ 2.0 + 0.5
        ori_list[:,1]= ori_list[:,1]/ 2.0 + 0.5
        ori_list = ori_list[:,[2,1,0]]
        segments = (np.ones(int(len(strand)/100))*100).astype("int")
        ori_list = ori_list.reshape((-1,100,3))
        _,depth,img2 = render_strand(strand,segments,self.mesh,orientation=ori_list,intensity=3,mask=True)
        oriImg1 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        mask_area = np.where(oriImg1!=0)
        mask = np.zeros_like(oriImg1)
        mask[mask_area]=255
        image = np.copy(img2)
        image = get_conditional_input_data1(image, mask, False, True)
        # flip=True
        if flip:
            image=(image-0.5)*2
            image = image[:, ::-1, :] * np.array([1.,1.,-1])
            image=(image+1)/2
            mask = mask[:, ::-1]
            image[mask!=255]=[0,0,0]
            depth = depth[:,::-1]
        # cv2.imwrite("1-2.png",(255*image).astype('uint8'))
        if not self.opt.no_use_bust: 
            oriImg1=np.zeros_like(mask)
            oriImg1[mask!=255]=1
            image[:,:,2]=image[:,:,2]+oriImg1*depth
            image[:,:,1]=image[:,:,1]+oriImg1*depth

        image = image[:, :, [2, 1]].astype(np.float32)
        image=torch.from_numpy(image)
        image=image.permute(2,0,1)
        Ori2D = image.clone()
        # save_image(torch.cat([Ori2D.unsqueeze(0), torch.zeros(1, 1, 256, 256)], dim=1)[:, :3, ...], 'test.png')

        data_list['image']=image
        if self.opt.use_HD:
            add_info=get_add_info(file_name,self.opt.strand_size,self.opt.info_mode,use_gt=True)
            # strand2D=get_strand2D(file_name,self.opt.strand_size,self.opt.strand_mode,use_gt)
            if self.opt.info_mode!='L':
                add_info=torch.from_numpy(add_info)
                add_info=add_info.permute(2,0,1)
            data_list['add_info']=add_info

        depth=cv2.resize(depth,(self.opt.image_size,self.opt.image_size))
        depth=depth[:,:,None]*95.0
        if self.opt.input_nc==3:
            mask1=np.zeros_like(mask)
            area=np.where((mask>0) & (depth[:, :, 0]>0))
            mask1[area]=1
            depth_masked=depth[:, :, 0] * mask1 - (1 - mask1) * (np.abs(np.nanmax(depth)) + np.abs(np.nanmin(depth)))
            max_val = np.nanmax(depth_masked)
            min_val = np.nanmin(depth_masked + 2 * (1 - mask1) * (np.abs(np.nanmax(depth)) + np.abs(np.nanmin(depth))))
            depth_norm = (max_val-depth_masked) / (max_val - min_val)*mask1
            depth_norm = np.clip(depth_norm, 0., 1.)
            # cv2.imwrite("depth_norm.png",(depth_norm*255).astype('uint8'))
            depth_norm=torch.from_numpy(depth_norm[:,:,None])
            depth_norm=depth_norm.permute(2,0,1)
            image=torch.cat([image,depth_norm], dim=0)
            data_list['image']=image
            # save_image(image, 'depth.png')
        depth=torch.from_numpy(depth)
        depth=depth.permute(2,0,1)
        data_list['depth']=depth

        gt_orientation = get_ground_truth_3D_ori1(file_name, ang, np.copy((image.numpy().transpose(1,2,0)*255).astype('uint8')), flip, growInv=self.opt.growInv)
        # image,gt_orientation,strand2D=self.random_translation(image,gt_orientation,strand2D)
        # gt_orientation 96,128,128,3
        gt_orientation,data_list=self.random_translation(self.opt.image_size,gt_orientation,data_list)


        mask=np.linalg.norm(gt_orientation,axis=-1)
        gt_occ=(mask>0).astype(np.float32)[...,None]
        gt_orientation=torch.from_numpy(gt_orientation)
        gt_occ=torch.from_numpy(gt_occ)
        gt_orientation=gt_orientation.permute(3,0,1,2)#3,96,128,128
        gt_occ=gt_occ.permute(3,0,1,2)#1,96,128,128

        if 'add_info' in data_list:
            add_info=data_list['add_info']
        else:
            add_info=torch.Tensor(0)
        return_list={
            'gt_ori':gt_orientation,
            'image':data_list['image'],
            'gt_occ':gt_occ,
            'Ori2D':Ori2D,
            'add_info':add_info,
            'depth':data_list['depth']
        }
        if self.opt.no_use_depth==False:
            return_list['norm_depth']=data_list['norm_depth']
        return return_list

    def generate_test_data(self):
        path = os.path.join(self.root, self.opt.test_file)
        save_path=os.path.join(self.opt.current_path, self.opt.save_root, self.opt.check_name, 'record', self.opt.test_file)
        if not os.path.exists(save_path):
            mkdir(save_path)
        # orismooth=cv2.imread(os.path.join(path,'Ori.png'))
        # orismooth=cv2.resize(orismooth,(1024,1024))
        # padding=np.zeros((4*8,1024,3))
        # orismooth=np.concatenate([padding,orismooth],axis=2)[:1024]
        # cv2.imwrite(os.path.join(save_path,'OriSmooth2D.png'),orismooth)
        # shutil.copyfile(os.path.join(path,'Ori.png'),os.path.join(save_path,'OriSmooth2D.png'))
        image = get_image(path, False, self.opt.image_size, self.opt.Ori_mode,self.opt.blur_ori,self.opt.no_use_depth,use_gt=False)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        Ori2D=image.clone()
        if not self.opt.no_use_L:
            print('use L map')
            L_map = get_luminance_map(path, False, self.opt.image_size,self.opt.no_use_bust)
            image=torch.cat([image,L_map],0)
        elif not self.opt.no_use_bust:
            image = get_Bust1(self.opt.current_path, image, self.opt.image_size)
        if os.path.exists(path+'/Ori_gt.mat'):
            gt_orientation = get_ground_truth_3D_ori(path, False, growInv=self.opt.growInv)
            mask = np.linalg.norm(gt_orientation, axis=-1)
            gt_occ = (mask > 0).astype(np.float32)[..., None]
            gt_orientation = torch.from_numpy(gt_orientation)
            gt_occ = torch.from_numpy(gt_occ)
            gt_orientation = gt_orientation.permute(3, 0, 1, 2)
            gt_occ = gt_occ.permute(3, 0, 1, 2)

            gt_occ=torch.unsqueeze(gt_occ,0)
            gt_orientation=torch.unsqueeze(gt_orientation,0)

        else:
            gt_orientation=torch.Tensor(0)
            gt_occ=torch.Tensor(0)
        image = torch.unsqueeze(image, 0)
        Ori2D=torch.unsqueeze(Ori2D,0)

        return_list = {
            'gt_ori': gt_orientation,
            'image': image,
            'gt_occ': gt_occ,
            'Ori2D':Ori2D
        }
        return return_list


    def generate_random_root(self,ori):
        # occ=torch.norm(ori,p=2,dim=1,keepdim=True)
        # occ=(occ>0).type(torch.float32)
        occ=np.linalg.norm(ori,axis=-1)[0]
        occ=(occ>0).astype(np.float32)
        samle_voxel_index =np.where(occ>0)
        samle_voxel_index=np.array(samle_voxel_index)
        samle_voxel_index=samle_voxel_index.transpose(1,0)
        random_points=samle_voxel_index[np.random.randint(0,samle_voxel_index.shape[0]-1,size=self.opt.num_root)]
        random_points=random_points[:,::-1]+np.random.random(random_points.shape[:])[None]
        random_points=random_points[...,None,:]
        random_points=torch.from_numpy(random_points)
        random_points=torch.reshape(random_points,(len(self.opt.gpu_ids),-1,1,3))

        return random_points



    def translation(self,data,offset_x,offset_y,mul,shape,rand_x,rand_y):
        padding_x=torch.zeros(shape[0],offset_x*mul,shape[1])
        padding_y=torch.zeros(shape[0],shape[1],offset_y*mul)
        if rand_x<0.5:
            data=torch.cat([data,padding_x],dim=1)[:,offset_x*mul:,:]
        else:
            data = torch.cat([padding_x, data], dim=1)[:, :-offset_x*mul, :]
        if rand_y<0.5:
            data = torch.cat([data, padding_y], dim=2)[:, :, offset_y*mul:]
        else:
            data = torch.cat([ padding_y,data], dim=2)[:, :, :-offset_y*mul]
        return data

    def random_translation(self, image_size, gt, data_list):
        offset_x = random.randint(1, 10) * 2
        offset_y = random.randint(1, 10) * 2
        rand_x = random.random()
        rand_y = random.random()

        for k,v in data_list.items():
            C,H,W=v.shape[:]
            mul=H//image_size
            data_list[k]=self.translation(v,offset_x,offset_y,mul,[C,H,W],rand_x,rand_y)

        gt_depth, gt_x, gt_y, gt_channel = gt.shape[:]

        padding_gt_x = torch.zeros(gt_depth, offset_x // 2, gt_y, gt_channel)
        padding_gt_y = torch.zeros(gt_depth, gt_x, offset_y // 2, gt_channel)
        if rand_x < 0.5:
            gt = np.concatenate([gt, padding_gt_x], axis=1)[:, offset_x // 2:, :, :]
        else:
            gt = np.concatenate([padding_gt_x, gt], axis=1)[:, :-offset_x // 2, :, :]
        if rand_y < 0.5:
            gt = np.concatenate([gt, padding_gt_y], axis=2)[:, :, offset_y // 2:, :]
        else:
            gt = np.concatenate([padding_gt_y, gt], axis=2)[:, :, :-offset_y // 2, :]
        return gt,data_list