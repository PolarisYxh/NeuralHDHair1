#for cvpr 2023 hair_step
from dataload.base_loader import base_loader
from Tools.utils import *
import torch
from Loss.percepture_loss import VGGLoss 
import shutil
import torchvision
class origin_step_loader(base_loader):

    def initialize(self,opt):

        self.opt = opt
        self.batch_size = opt.batch_size
        self.image_size = opt.image_size
        self.isTrain = opt.isTrain
        self.parent_dir = opt.current_path
        self.root = os.path.join(self.parent_dir,opt.strand_dir)
        if self.isTrain:
            self.train_corpus = []#变量名不能随便改，定义在base_loader里面了
            self.train_nums = 0
            self.generate_corpus()
        else:
            self.train_corpus = []
            self.generate_corpus(False)

    def generate_corpus(self,is_train=True):
        # exclude the tail, to do improve this
        if is_train:
            self.datas = read_json(os.path.join(self.root,"split_train1.json"))
            # self.datas = list(filter(lambda a:f"XH" in a and "004" not in a, self.datas))
            print(f"train images data length:{len(self.datas)}")
        else:
            self.datas = read_json(os.path.join(self.root,"split_test1.json"))
            print(f"test images data length:{len(self.datas)}")
        device = torch.device("cuda") if torch.cuda.is_available() and len(self.opt.gpu_ids)>0 else torch.device("cpu")
        crit_vgg = VGGLoss(model='vgg19', gpu_ids = self.opt.gpu_ids, layer=35)
        random.shuffle(self.datas)
        for x in self.datas[:]:
            target=cv2.imread(os.path.join(self.root,"strand_map",x))#R：255,B:（0,1）表示（向右，向左）；G：第二通道，（0,1）表示（向上，向下）
            # TODO:两种方式得到的segment图不太一样，哪个比较好 后续进行实验
            target0=cv2.resize(target, (512, 512))# TODO: size大小到底是多少
            # target0=target
            parse = target0[:, :, 2]
            M_sum = torch.tensor(len(np.where(parse>200)[0]))
            
            target0[np.where(parse<200)]=[0,0,0]
            target0=target0[:,:,[1,0,2]]
            target0=target0.transpose([2,0,1])
            target0=torch.from_numpy(target0)[None] / 255
            
            # target0=255-target0
            # target0[np.where(parse<200)]=[0,0,0]
            # # target0=cv2.cvtColor(target0,cv2.COLOR_RGB2BGR)
            # # target0=target
            # # cv2.imshow("1",target0)
            # # cv2.waitKey()
            # target0=target0.transpose([2,0,1])
            # target0=torch.from_numpy(target0)[None] / 255
            # M_sum = torch.sum(target0[:,2,:,:])
            target0 = target0.to(device)
            target_act = crit_vgg.get_features(target0)
            
            data = {"gt_feat":target_act.to('cpu').squeeze(),"gt_sum":M_sum}
            self.train_corpus.append(data)
            
        self.train_nums = len(self.train_corpus)
        # print('val strand:',self.val_corpus)
        # print('train strand:',self.train_corpus)
        print(f"num of training data: {self.train_nums}")

    def __getitem__(self, index):
        x=self.datas[index]
        data_list={}
        input=cv2.imread(os.path.join(self.root,"img",x))
        input= cv2.resize(input,(512,512))
        seg =  cv2.imread(os.path.join(self.root,"seg",x))
        seg= cv2.resize(seg,(512,512))
        seg[seg<127]=0
        seg[seg>=127]=1
        input = input*seg
        
        input=input.transpose([2,0,1])
        input=torch.from_numpy(input) / 255
        # torchvision.utils.save_image(input.unsqueeze(0).cpu(),"test3.png")
        target0=cv2.imread(os.path.join(self.root,"strand_map",x))
        target0=cv2.resize(target0, (512, 512))# TODO: size大小到底是多少
        parse = target0[:, :, 2]
        target0[np.where(parse<200)]=[0,0,0]
        target0=target0[:,:,[1,0]]
        target0=target0.transpose([2,0,1])
        target0=torch.from_numpy(target0) / 255
        seg=torch.from_numpy(seg).permute([2,0,1])
        data_list['input']=input
        data_list['target']=target0
        data_list['seg']=seg
        data_list=self.random_translation(512,data_list)
        if index==0:
            save_image(data_list['input'],"i1.png")
            save_image(data_list['target'],"i2.png")
        return {"seg":data_list['seg'],"input":data_list['input'],"target":data_list['target'],"gt_feat":self.train_corpus[index]["gt_feat"],"gt_sum":self.train_corpus[index]["gt_sum"]}

    def generate_test_data(self):
        path = os.path.join(self.root, self.opt.test_file)
        save_path=os.path.join(self.opt.current_path, self.opt.save_root, self.opt.check_name, 'record', self.opt.test_file)
        if not os.path.exists(save_path):
            mkdir(save_path)
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

    def random(self, image_size, gt, data_list):
        from torchvision import transforms
        # 创建一个图像变换对象，包含随机旋转、缩放和平移
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=(-180,180),translate=(0.0,0.6),scale=(1,2)),
            transforms.ToTensor()  # 转换为张量
        ])

        # 加载图像
        image = Image.open('path/to/image.jpg')

        # 应用图像变换
        transformed_image = transform(image)
        return gt,data_list
    
    def random_translation(self, image_size, data_list):
        offset_x = random.randint(1, 20) * 2
        offset_y = random.randint(1, 20) * 2
        rand_x = random.random()
        rand_y = random.random()

        for k,v in data_list.items():
            C,H,W=v.shape[:]
            mul=H//image_size
            data_list[k]=self.translation(v,offset_x,offset_y,mul,[C,H,W],rand_x,rand_y)

        
        return data_list






