import os
import sys
from pyhocon import ConfigFactory
from pathlib import Path

import torch

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras,FoVOrthographicCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform,look_at_rotation
from pytorch3d.io import load_ply, save_ply

sys.path.append(os.path.join(sys.path[0], '..'))
# from src.models.dataset import Dataset, MonocularDataset

import argparse
import yaml

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from skimage import transform
import numpy as np
import cv2
import trimesh
from src.utils.util import align_pyrender_camera

def create_visibility_map(camera, rasterizer, mesh):
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face  # giving the indices of the nearest faces at each pixel
    packed_faces = mesh.faces_packed() 
    packed_verts = mesh.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]) 
    faces_visibility_map = torch.zeros(packed_faces.shape[0]) 
    visible_faces = pix_to_face.unique()[1:] # not take -1
    visible_verts_idx = packed_faces[visible_faces] 
    unique_visible_verts_idx = torch.unique(visible_verts_idx)
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    return vertex_visibility_map, faces_visibility_map
import numba
from numba.typed import List
from numba.typed import Dict
from numba import types
@numba.njit
def get_faces(faces,idx,face_idx):
    face_faces=List()
    indices_mapping = Dict.empty(
        key_type=types.int64,  # numba_dict supports tuple types
        value_type=types.int64
    )
    for i, j in enumerate(idx):
        indices_mapping[j] = i 
    for fle, i in enumerate(faces):#full_mesh.faces_packed()：所有face的三个顶点索引N*3，i：某个面的三个顶点
        if i[0] in face_idx and i[1] in face_idx and i[2] in face_idx:#面上三个点都可见
            face_faces.append([indices_mapping[i[0]], indices_mapping[i[1]], indices_mapping[i[2]]])
            
    return face_faces

def check_visiblity_of_faces(cams, meshRasterizer, full_mesh, mesh_head, n_views=2):
    # collect visibility maps
    vis_maps = []
    for cam in tqdm(range(len(cams))):
        v, _ = create_visibility_map(cams[cam], meshRasterizer, full_mesh)
        idx = torch.nonzero(v)
        vis_maps.append(v)

    # took faces that were visible at least from n_views to reduce noise
    vis_mask = (torch.stack(vis_maps).sum(0) > n_views).float()

    idx = torch.nonzero(vis_mask).squeeze(-1).tolist()
    
    
    # idx = [i for i in idx if i > mesh_head.verts_packed().shape[0]]#可见的头发的顶点的索引
    indices_mapping = {j: i for i, j in enumerate(idx)}#{顶点的全身索引：顶点的可见模型索引}

    
    face_idx = torch.tensor(idx).to('cuda')#可见的头发的顶点的索引
    vertex = full_mesh.verts_packed()[face_idx]
    # import time
    # start_time = time.time()
    face_idx = face_idx.to('cpu').numpy()
    faces = full_mesh.faces_packed()
    faces = faces.to('cpu').numpy()
    face_faces = get_faces(faces,idx,face_idx)
    face_faces=np.array(face_faces)
    # end_time = time.time()
    # print(f"Function took {(end_time - start_time):.8f} seconds to execute.")
    # origin code
    # import time
    # start_time = time.time()
    # face_faces = []
    # # face_idx = idx
    # face_idx = torch.tensor(idx).to('cuda')
    # faces = full_mesh.faces_packed()
    # for fle, i in enumerate(faces):
    #     if i[0] in face_idx and i[1] in face_idx and i[2] in face_idx:
    #         face_faces += [[indices_mapping[i[0].item()], indices_mapping[i[1].item()], indices_mapping[i[2].item()]]]
    # end_time = time.time()
    # print(f"Function took {(end_time - start_time):.8f} seconds to execute.")
    return vertex, torch.tensor(face_faces)
def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
# def align_pyrender_camera(pos=None,device='cpu'):
#     # pytorch3d相机对齐pyrender相机得到RT，法一，使用相机位姿矩阵(从image_filter.py得到self.cam_pose = x2@four_by_four@x1@np.linalg.inv(m[2]))
#     if not isinstance(pos,np.ndarray):
#         pos = np.array([[ 9.61313906e-01,  7.01898160e-02, -2.66362467e-01, -7.95505687e-02],
#                         [ 6.93889390e-18,  9.66990113e-01,  2.54813897e-01, 1.65589804e+00],
#                         [ 2.75455212e-01, -2.44956142e-01,  9.29581042e-01, 2.64301285e-01],
#                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])# for test
#     cam_T=torch.from_numpy(pos[:3,3])[None].to(torch.float32)
#     R = pos[:3,:3]
#     rot_matrix = transform.SimilarityTransform(rotation=[0,np.deg2rad(180),0],dimensionality=3).params[:3,:3]
#     R = torch.from_numpy(R@rot_matrix).to(torch.float32)[None]#绕y轴旋转180度，即第一第三列取反
#     T = -torch.bmm(R.transpose(1, 2), cam_T[:, :, None])[:, :, 0]
#     # pytorch3d相机对齐pyrender相机得到RT，法二，使用目标位置和相机位置T
#     # pos1 = np.array([[ 9.61313906e-01,  7.01898160e-02, -2.66362467e-01, -7.95505687e-02],
#     #                 [ 6.93889390e-18,  9.66990113e-01,  2.54813897e-01, 1.65589804e+00],
#     #                 [ 2.75455212e-01, -2.44956142e-01,  9.29581042e-01, 2.64301285e-01],
#     #                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
#     # tar_pose = np.array([0, 1.58652416, 0])#NeuralHDHair默认的
#     # tar_pose = torch.from_numpy(tar_pose)[None].to(torch.float32)
#     # cam_T=torch.from_numpy(pos1[:3,3])[None].to(torch.float32)
#     # R0, T0 = look_at_view_transform(eye=torch.from_numpy(pos1[:3,3])[None].to(torch.float32),at=tar_pose)
#     # look_at_view_transform
#     # R1 = look_at_rotation(cam_T, tar_pose) ==R0
#     # T1 = -torch.bmm(R1.transpose(1, 2), cam_T[:, :, None])[:, :, 0] ==T0
    
#     # camera intrinsic
#     cam = 0.28347224#96*0.00567194(voxel_size)-0.261034
#     far = 0.261034
#     xymag = 0.36300416
#     cams = [FoVOrthographicCameras(device=device,znear=0.0001,zfar=cam+far,R=R,T=T,max_x=xymag,min_x=-xymag,max_y=xymag,min_y=-xymag)]
#     return cams

class ExtractVisibleSurface:
    def __init__(self,device='cuda') -> None:
        self.device=device
        # add upper cameras for h3ds data as they haven't got such views
        self.cams_up = []
        #angs:第一维为azim(先水平方向旋转),第二维是elev(后竖直方向旋转)；
        #azim=90:elevs=0表明相机从人体右边往人体看，elevs=45表明相机从人体右上向人体看;azim=0:elevs=45表明相机从人体前上朝人体看
        angs = [[0,0],[0,45],[0,90],[0,135],[0,180],[0,225],[0,-45],[0,-90],\
                [90,-45],[90,0],[90,45],[90,135],[90,180],[90,225]]
        
        for ang in angs:
            R, T = look_at_view_transform(dist=4, elev=ang[1], azim=ang[0])
            cam = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            self.cams_up.append(cam)
        self.body = trimesh.load_mesh("./female_halfbody_medium_join.obj")
    
    def get_outer_faces(self, verts, faces,img_size=1024):
        mesh_hair =  Meshes(verts=[(verts).float().to(self.device)], faces=[faces.to(self.device)])
        
        mesh_head =  Meshes(verts=[(torch.from_numpy(self.body.vertices)).float().to(self.device)], faces=[torch.from_numpy(self.body.faces).to(self.device)])
        raster_settings_mesh = RasterizationSettings(
                            image_size=img_size, 
                            blur_radius=0.000, 
                            faces_per_pixel=1, 
                            max_faces_per_bin=0
                        )
        R = torch.ones(1, 3, 3)
        t = torch.ones(1, 3)
        cam_intr = torch.ones(1, 4, 4)
        size = torch.tensor([img_size, img_size ]).to(self.device)

        cam = cameras_from_opencv_projection(
                                            camera_matrix=cam_intr.cuda(), 
                                            R=R.cuda(),
                                            tvec=t.cuda(),
                                            image_size=size[None].cuda()
                                            ).cuda()

        # init mesh rasterization
        meshRasterizer = MeshRasterizer(cam, raster_settings_mesh)

        mesh_hair.textures = TexturesVertex(verts_features=torch.ones_like(mesh_hair.verts_packed()).float().cuda()[None])
        mesh_head.textures = TexturesVertex(verts_features=torch.zeros_like(mesh_head.verts_packed()).float().cuda()[None])

        # join hair and bust mesh to handle occlusions
        full_mesh = join_meshes_as_scene([mesh_head, mesh_hair])
        # full_mesh = mesh_head
        # cams_dataset = [cameras_from_opencv_projection(
        #                                 camera_matrix=self.intrinsics_all[idx][None].cuda(), 
        #                                 R=self.pose_all_inv[idx][:3, :3][None].cuda(),
        #                                 tvec=self.pose_all_inv[idx][:3, 3][None].cuda(),
        #                                 image_size=size[None].cuda()
        #                                 ).cuda() for idx in range(len(self.pose_all_inv))]

        cams = self.cams_up #cams_dataset #+ 
        intri = torch.tensor([[ 2.75478937,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  2.75478937,  0.        ,  0.        ],
       [ 0.        ,  0.        , -3.67372718, -1.00036737],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])[None]
        cams = align_pyrender_camera(device=self.device,intri=intri)
        
        vis_vertex, vis_face = check_visiblity_of_faces(cams, meshRasterizer, full_mesh, mesh_head, n_views=0)

        mesh_hair_outer =  Meshes(verts=[(vis_vertex).float().to(self.device)], faces=[vis_face.to(self.device)])
        save_ply('hair_outer.ply', vis_vertex, vis_face)
        return mesh_hair_outer

    
if __name__=="__main__":
    R, T = look_at_view_transform(dist=4, elev=0, azim=0)
    cam = FoVPerspectiveCameras(R=R, T=T)
    
    save_dir = "/nvme0/yangxinhang/HairStrand/NeuralHDHairData"
    names = os.listdir(save_dir)
    import trimesh
    from skimage import measure
    import torch.nn.functional as F
    from Tools.utils import transform_Inv
    import mcubes
    body = trimesh.load_mesh("./female_halfbody_medium_join.obj")
    
    for file in names[3:10]:# 'strands00127'
        file = 'strands00065'
        ori=np.load(os.path.join(save_dir,file,'Ori_gt_hg.npy'),allow_pickle=True)
        
        ori = np.load("/home/algo/yangxinhang/NeuralHDHair/NeuralHaircut/test/img_0044/img_0044_orientation.npy",allow_pickle=True)
        ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
        ori = ori.transpose([0, 1, 3, 2])
        mask=np.linalg.norm(ori.reshape((ori.shape[0], ori.shape[1], -1, 3)),axis=-1)
        gt_occ=(mask>0).astype(np.float32)
        
        gt_occ = F.max_pool3d(torch.from_numpy(gt_occ[...,None]), kernel_size=5, stride=1, padding=2)
        # gt_occ_erode = 1-F.max_pool3d(1-gt_occ, kernel_size=11, stride=1, padding=5)
        
        # gt_occ = gt_occ-gt_occ_erode
        gt_occ=gt_occ.numpy()[...,0]
        
        verts, faces, normals, values = measure.marching_cubes(gt_occ.transpose((1,0,2)), 0.5)
        verts, faces = mcubes.marching_cubes(gt_occ.transpose((1,0,2)), 0.5)
        verts = transform_Inv(verts,scale=ori.shape[0]//128)
    
        hair_mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
        hair_mesh = trimesh.smoothing.filter_laplacian(hair_mesh, iterations=20)
        hair_mesh.export(f"{file}.obj")
        e = ExtractVisibleSurface()
        e.get_outer_faces(torch.from_numpy(hair_mesh.vertices),torch.from_numpy(hair_mesh.faces))        
