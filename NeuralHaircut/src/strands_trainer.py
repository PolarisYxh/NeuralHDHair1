import torch
import torch.nn.functional as F
import yaml
from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from src.hair_networks.strands_renderer import Renderer
from src.losses.sdf_chamfer import points2face,calc_chamfer
import os
import torch.nn as nn
import numpy as np

from src.utils.util import freeze_gradients
from skimage import measure
import trimesh
from Tools.utils import transform_Inv
from preprocess_custom_data.extract_visible_surface import ExtractVisibleSurface
import mcubes
from Code.Tools.utils import mesh_to_voxel_torch,mesh_to_voxel
class StrandsTrainer:
    def __init__(self, config, run_model=None, device=None, save_dir=None) -> None:
        
        self.device = device
        
        params_to_train = []
        self.strands = OptimizableTexturedStrands(**config['textured_strands'], diffusion_cfg=config['diffusion_prior']).to(self.device)
        params_to_train += list(self.strands.parameters())

        self.strands_render = None
        if config['render']['use_render']:
            print('Create rasterizer!')
            self.strands_render = Renderer(config['render'], save_dir=save_dir).to(self.device)
            params_to_train += list(self.strands_render.parameters())

        self.starting_rendering_iter = config['general']['starting_rendering_iter']  

        self.optimizer = torch.optim.Adam(params_to_train, config['general']['lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['general']['milestones'], gamma=config['general']['gamma'])
        self.loss_factors = config['loss_factors']

        # self.sdfchamfer = SdfChamfer(**config['sdf_chamfer'])
        self.run_model = run_model
        self.strands_origins = None
        # self.extract_visible_face = ExtractVisibleSurface(device)
        from pytorch3d.io import IO
        self.mesh_outer_hair = IO().load_mesh('./NeuralHaircut/test/img_0044/hair_outer_0044.ply', device=self.device)
    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.strands.load_state_dict(state_dict['strands'])
        if self.strands_render is not None:
            self.strands_render.load_state_dict(state_dict['strands_render'])


    def save_weights(self, path):
        state_dict = {
            'strands': self.strands.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if self.strands_render is not None:
            state_dict['strands_render'] = self.strands_render.state_dict()
        torch.save(state_dict, path)

    def get_outer_hair(self,gt_occ):#ori:DHW3; gt_occ:DHW1
        
        gt_occ = F.max_pool3d(torch.from_numpy(gt_occ), kernel_size=5, stride=1, padding=2)
        # gt_occ_erode = 1-F.max_pool3d(1-gt_occ, kernel_size=11, stride=1, padding=5)
        
        # gt_occ = gt_occ-gt_occ_erode
        gt_occ=gt_occ.numpy()[...,0]#gt_occ:DHW
        
        verts, faces, normals, values = measure.marching_cubes(gt_occ.transpose((1,0,2)), 0.5)
        # verts, faces = mcubes.marching_cubes(gt_occ.transpose((1,0,2)), 0.5)
        
        verts = transform_Inv(verts,scale=gt_occ.shape[0]//128)
        
        # verts = verts+np.array([0.00703544,-1.58652416,-0.01121912])
        # verts = np.dot(verts, revert_rot)+np.array([-0.00703544,1.58652416,0.01121912])
        hair_mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
        hair_mesh = trimesh.smoothing.filter_laplacian(hair_mesh, iterations=20)
        # hair_mesh.export(f"1.obj")

        mesh_outer_hair = self.extract_visible_face.get_outer_faces(torch.from_numpy(hair_mesh.vertices),torch.from_numpy(hair_mesh.faces)) 
        return mesh_outer_hair
    def index_voxel(self,feat,uv):
        '''
        :param feat: [B, C, H, W] image features,(N, C, D, H_, W)(5-D case VOXEL)
        :param uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]!,and feat:xyzw; uv:wzyx 指的维度相反即可
        :return: [B, C, N] sampled pixel values
        '''
        if len(feat.shape)==5:
            uv=uv.unsqueeze(2).unsqueeze(2)#[B,N,1,1,2](5-D case VOXEL)
        elif len(feat.shape)==4:
            uv=uv.unsqueeze(2)#[B,N,1,2](4-D case VOXEL)
        samples=torch.nn.functional.grid_sample(feat, uv, mode='nearest', align_corners=True)
        return samples[...,0]
    
    def train_step(self, ori, occ,it=0, raster_dict=None):#ori:HWD3; gt_occ:HWD1
        losses = {}
        
        # mesh_outer_hair = self.get_outer_hair(occ) 
        
        self.strands_origins, z_geom, z_app, dif_dict = self.strands(it=it)#self.strands_origins 1900,100,3
        # for debug
        # import trimesh
        # cols = torch.cat((torch.rand(self.strands_origins.shape[0], 3).unsqueeze(1).repeat(1, 100, 1), torch.ones(self.strands_origins.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()           
        # trimesh.PointCloud(self.strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export('strands_points.ply')
        # from Code.dataload.render_strand import render_strand
        # import cv2
        # x=self.strands_origins.detach().to('cpu').numpy().reshape([-1,3])
        # segments = np.array(list(range(0,len(x),100)))
        # body = trimesh.load_mesh("female_halfbody_medium_join.obj")
        # _,_,r = render_strand(x,segments,body)
        # cv2.imwrite("1.png",r)
        strand_len = self.strands_origins.shape[1]
        # 
        # with freeze_gradients(model):#第二步得到的gt，#out:[190000,1],[190000, 256],[190000, 3]为sdf,feat,orient
        #     out = self.run_model(model, self.strands_origins.view(-1, 3))
        # self.strands_origins_voxel:[W,H,D]
        self.strands_origins_voxel = mesh_to_voxel_torch(self.strands_origins.reshape((-1,3)),scale=2)
        self.strands_origins_voxel = self.strands_origins_voxel.to(torch.int64)[0].permute(1,0)
        self.strands_origins_voxel = (self.strands_origins_voxel/torch.tensor([occ.shape[1],occ.shape[0],occ.shape[2]]).to('cuda')-0.5)*2
        self.strands_origins_voxel = self.strands_origins_voxel[:,[2,0,1]]#[D,W，H]
        
        occ=torch.from_numpy(occ.copy()).to(self.device).to(torch.int)
        ori=torch.from_numpy(ori.copy()).to(self.device).to(torch.float32)
        #occ[None].permute((0, 4, 1, 2, 3)):N, C, H, W, D
        occ_list=self.index_voxel(occ[None].to(torch.float32).permute((0, 4, 1, 2, 3)),self.strands_origins_voxel[None])[0,0]
        ori_list=self.index_voxel(ori[None].to(torch.float32).permute((0, 4, 1, 2, 3)),self.strands_origins_voxel[None])[0,...,0].permute((1,0))
        
        losses['volume'] = torch.sum(1-torch.abs(occ_list))
        # Calculate origin loss
        # sdf = out[..., 0].view(-1, strand_len)
        # sdf_inside = torch.relu(sdf[:, 1:])
        # losses['volume'] = F.mse_loss(sdf_inside, torch.zeros_like(sdf_inside))

        # Calculate orientation loss
        prim_dir = self.strands_origins[:, 1:] - self.strands_origins[:, :-1] # [N_strands, strand_len-1, 3]
        pred_dir = ori_list[..., -3:].view(-1, strand_len, 3)[:, :-1] # [N_strands, strand_len-1, 3]

        
        dist = points2face(self.mesh_outer_hair, self.strands_origins[:, :-1, :].reshape(-1, 3)) 
        filtered_idx = torch.nonzero(dist[0] <= 0.001).reshape(-1)
        losses['orient'] = (1 - torch.abs(torch.cosine_similarity(prim_dir.reshape(-1, 3)[filtered_idx], pred_dir.reshape(-1, 3)[filtered_idx], dim=-1))).mean()

        # Calculate chamfer on visible outer surface
        losses['chamfer'] = calc_chamfer(self.mesh_outer_hair,self.strands_origins[:, 1:, :].reshape(-1, 3)[None])

        # Calculate photometric losses
        if self.strands_render and it > self.starting_rendering_iter:
            raster_dict = self.strands_render( 
                                        strands_origins=self.strands_origins,
                                        z_app=z_app,
                                        raster_dict=raster_dict,
                                        iter=it,
                                         )

            raster_loss = self.strands_render.calculate_losses(raster_dict, it)

            if 'silh' in raster_loss.keys():
                losses['raster_silh'] = raster_loss['silh']
            if 'alpha_prediction' in raster_loss.keys():
                losses['raster_alpha'] = raster_loss['alpha_prediction']
            losses['raster_l1'] = raster_loss['l1']
        
        # Calculate diffusion loss
        if len(dif_dict) > 0:
            losses['L_diff'] = dif_dict['L_diff']

        self.optimizer.zero_grad()

        total_loss = sum(loss * float(self.loss_factors[name]) for name, loss in losses.items())
        total_loss.backward()

        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None and param.grad.isnan().any():
                self.optimizer.zero_grad()
                print('NaN during backprop was found, skipping iteration...')
                return losses
            
        self.optimizer.step()
        self.scheduler.step()

        return losses
