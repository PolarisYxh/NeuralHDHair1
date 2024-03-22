#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# 实现两个形状相似mesh的配准，即将mesh_s变成mesh_t的形状和颜色，而保留原拓扑
import logging
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import trimesh
from utils.utils import (apply_transform_matrix,
                                          find_closest_ids, getSubMesh,
                                          readjson, timeCost,
                                          transform_matrix2SRT,
                                          transform_SRT2matrix, writejson)
from utils.utils_glb import (merge_mesh_vertices, mix_glb_objs,
                                              show_diff_two_meshes,
                                              show_vertex_indices)
from trimesh.proximity import closest_point
from trimesh.registration import nricp_amberg, procrustes
from trimesh.triangles import points_to_barycentric
#from utils.utils_mix import load_resize_img, save_img
#from utils.utils_pytorch3d import cvt_texture_by_uv

bDebug = 0
bCache = True
bForce = 1

cacheDir = "cache/FaceRegister/"


class FaceRegister():
    def __init__(self, mesh_s, lms_index_s):
        self.mesh_s = mesh_s
        #转成没有空间重合点的网格
        self.mesh_s_m, self.ids_o2u_s, self.ids_u2o_s = merge_mesh_vertices(mesh_s, merge_tex=True, merge_norm=True)
        self.lms_index_s = list(np.array(self.ids_u2o_s)[lms_index_s])
        #show_vertex_indices(mesh_s_m,lms_index_s)
        self.use_barycentric_coordinates = True
        # Parameters for nricp_amberg
        wl = 3
        max_iter = 10
        wn = 0.5

        # ws is smoothness term,
        # wl weights landmark importance,
        # wn normal importance,
        # max_iter is the maximum number of iterations per step.
        self.steps_amberg0 = [
            # ws, wl, wn, max_iter
            [0.02, wl, wn, max_iter],
            [0.007, wl, wn, max_iter],
            [0.002, wl, wn, max_iter],
        ]
        self.steps_amberg = [
            # ws, wl, wn, max_iter
            [0.02, wl * 0.8, wn * 0.5, max_iter],
            [0.007, wl * 0.4, wn * 0.5, max_iter],
            [0.002, wl * 0.1, wn * 0.5, max_iter],
        ]

        self.distance_threshold = 0.05

    @timeCost
    def process_kernel(self, mesh_s, lms_index_s0, mesh_t, lms_t, name="", bRicp=False):
        #配置权重
        mask_lms = np.ones(len(lms_index_s0))
        if 0:
            #眼睛
            mask_lms[17:25] = 10
            mask_lms[69:77] = 10
            #嘴唇
            mask_lms[34:54] = 10
            #眼角
            mask_lms[18] = 200
            mask_lms[21] = 50
            mask_lms[70] = 50
            mask_lms[73] = 100
            #下巴
            mask_lms[8] = 50
            #嘴角等
            mask_lms[34] = 50
            mask_lms[43] = 50
            mask_lms[35] = 50
            mask_lms[53] = 50

            lms_index_s = []
            lms_t_weight_id = []
            for i in range(mask_lms.shape[0]):
                for j in range(int(mask_lms[i])):
                    lms_index_s.append(lms_index_s0[i])
                    lms_t_weight_id.append(i)
        
            lms_s = mesh_s.vertices[lms_index_s]
            lms_t = lms_t[lms_t_weight_id]
        else:
            lms_index_s = lms_index_s0
            lms_s = mesh_s.vertices[lms_index_s0]
        T = procrustes(lms_s, lms_t)[0]
        #trans, rotation, scale=transform_matrix2SRT(T)
        mesh_s.apply_transform(T)
        if bDebug and 1:
            sce = trimesh.Scene()
            sce.add_geometry(mesh_s)
            sce.add_geometry(mesh_t)
            sce.show()
        if bRicp:
            return mesh_s.vertices, T
        if self.use_barycentric_coordinates:
            source_markers_tids = closest_point(mesh_s, lms_s)[2]
            source_markers_barys = points_to_barycentric(mesh_s.triangles[source_markers_tids], lms_s)
            source_landmarks = (source_markers_tids, source_markers_barys)
        else:
            source_landmarks = lms_index_s

        records = nricp_amberg(mesh_s,
                               mesh_t,
                               source_landmarks=source_landmarks,
                               distance_threshold=self.distance_threshold,
                               target_positions=lms_t,
                               steps=self.steps_amberg,
                               return_records=bDebug)

        if bDebug:
            os.makedirs(cacheDir, exist_ok=True)
            logging.info(f"step of nricp_amberg = {len(records)-1}")
            records=[records[-1]]
            if False:
                mesh_s.vertices = records[0]
                mesh_s.export(cacheDir + f'{name}st.obj')
                img = show_diff_two_meshes(mesh_t, mesh_s, text="", bSameOrder=False, interactive=False, img_size=[640, 640], camera_distance=10)
                Image.fromarray(img).save(cacheDir + 'result_nricp_st.png')
                mesh_s.vertices = records[-1]
                mesh_s.export(cacheDir + f'{name}end.obj')
                img = show_diff_two_meshes(mesh_t, mesh_s, text="", bSameOrder=False, interactive=False, img_size=[640, 640], camera_distance=10)
                Image.fromarray(img).save(cacheDir + 'result_nricp_end.png')

            else:
                import pyvista as pv
                name = 'Amberg et al. 2007'
                distances = [closest_point(mesh_t, r)[1] for r in records]
                p = pv.Plotter()
                p.background_color = 'w'
                pv_mesh = pv.wrap(mesh_s)
                pv_mesh['distance'] = distances[0]
                p.add_text(name, color=(0, 0, 0))
                p.add_mesh(pv_mesh, color=(0.6, 0.6, 0.9), cmap='rainbow', clim=(0, mesh_t.scale / 100), scalars='distance', scalar_bar_args={'color': (0, 0, 0)})
                p.add_mesh(pv.wrap(mesh_t), style='wireframe')

                def cb(value):
                    t1 = min(int(value), len(records) - 1)
                    t2 = min(t1 + 1, len(records) - 1)
                    t = value - t1

                    pv_mesh.points = (1 - t) * records[t1] + t * records[t2]
                    for i, pos in enumerate(pv_mesh.points[lms_index_s]):
                        p.add_mesh(pv.Sphere(mesh_t.scale / 1000, pos), name=str(i), color='r')
                    pv_mesh['distance'] = (1 - t) * distances[t1] + t * distances[t2]

                p.add_slider_widget(cb, rng=(0, len(records)), value=0, color='black', event_type='always', title='step')

                for pos in lms_t:
                    p.add_mesh(pv.Sphere(mesh_t.scale / 1000, pos), color='g')
                p.show()
            vertices_r = records[-1]
        else:
            vertices_r = records
        return vertices_r, T

    def wrap_shape(self, mesh_t, lms_t, name, bRicp=False):
        mesh_t_m, ids_o2u_t, ids_u2o_t = merge_mesh_vertices(mesh_t, merge_tex=True, merge_norm=True)
        vertices_r_m, T = self.process_kernel(self.mesh_s_m, self.lms_index_s, mesh_t_m, lms_t, name=name, bRicp=bRicp)
        vertices_r = vertices_r_m[self.ids_u2o_s]
        return vertices_r, T


class FaceRegiterCase():
    def __init__(self, workingDir):
        file_mesh_s = workingDir + f'model/HMH/head_morph_shape526.glb'
        file_json_s = workingDir + "model/HMH/head_index.json"
        self.mesh_s = trimesh.load(file_mesh_s, force="mesh", process=False)
        self.head_index_s = readjson(file_json_s)
        self.lms_index_s = self.head_index_s["lms"]
        self.faceRegister = FaceRegister(self.mesh_s, self.lms_index_s)
        os.makedirs(workingDir+"cache/FaceRegister/",exist_ok=True)
        
        
    @timeCost
    def process_file(self,file_mesh_t):
        mesh_t = trimesh.load(file_mesh_t, force="mesh", process=False)
        mesh_r = self.mesh_s.copy()
        return mesh_r
# mesh放一起的可视化
def show_meshes_sidebyside(meshes, interactive=True, camera_position=(0.0, 0.0, 1.0), up=(0, 1, 0), focal_point=(0.0, 0.0, 0.0), view_angle=20.0):
    import pyvista as pv
    pl = pv.Plotter(shape=(1, len(meshes)))
    for i in range(len(meshes)):
        mesh = meshes[i]
        # mesh = pv.read(file0)
        # mesh.show()
        pl.subplot(0, i)
        try:
            pl.add_mesh(mesh, texture=pv.numpy_to_texture(np.array(get_texture_from_material(mesh.visual.material))))
        except:
            pl.add_mesh(mesh)

    pl.link_views()
    pl.camera_position = camera_position
    pl.camera.up = up
    pl.camera.focal_point = focal_point
    pl.camera.view_angle = view_angle
    pl.add_axes()
    # pl.show()
    img = pl.show(interactive=interactive, return_img=True)
    if 0:
        camera = pl.camera
        print("Camera Position:", camera.position)
        print("Camera Up:", camera.up)
        print("Camera Focal Point:", camera.focal_point)
    return img  
    
if __name__ == "__main__":
    import os
    import sys
    workingDir = os.path.split(sys.argv[0])[0]
    if (workingDir):
        workingDir = workingDir + "/"
    else:
        workingDir = "./"
    logging.basicConfig(
        level=logging.INFO,
        #filename=f"{workingDir}../log/fitter.log",
        format="[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        filemode='a',
        datefmt='%Y-%m-%d %A %H:%M:%S',
    )

    faceRegiterCase = FaceRegiterCase(workingDir)
    file_avatar = workingDir + "data/results/mesh.obj"
    faceRegiterCase.process_file(file_avatar)