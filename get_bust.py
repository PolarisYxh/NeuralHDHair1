import trimesh
import numpy as np
import cv2
from copy import deepcopy
from skimage import transform
import pyrender
import json
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict  
def render(tri_sence, preview_file="", bOrtho=False, intensity=2, matrix=[]):
    camera_pose_base = np.array([
    [1.0, 0.0, 0.0, 0],
    [0.0, 1.0, 0.0, 1.58652416],
    [0.0, 0.0, 1.0, 1.3],
    [0.0, 0.0, 0.0, 1.0],
    ])
    camera_pose = camera_pose_base
    tri_geometry = tri_sence
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1],bg_color=[0,0,0])
    bGlasses = False
    mesh = tri_sence

    mesh = pyrender.Mesh.from_trimesh(tri_geometry)
    scene.add(mesh)

    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=640, point_size=1.0)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(light, light_pose)
    if bOrtho:
        xymag = 0.36300416
        pc = pyrender.OrthographicCamera(xymag, xymag, zfar=100)
    else:
        pc = pyrender.PerspectiveCamera(yfov=13.3493 * np.pi / 180, aspectRatio=15/13.3493, znear=0.01,zfar=20)
    # print(pc.get_projection_matrix())
    scene.add(pc, pose=camera_pose)
    color, depth = r.render(scene, flags=flags)
    matrix.append(np.dot(pc.get_projection_matrix(), np.linalg.inv(scene.main_camera_node.matrix)))
    # print(matrix)
    img = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
    r.delete()
    if preview_file != "":
        cv2.imwrite(preview_file, img)
    return img
if __name__=="__main__":
    body = trimesh.load_mesh("/home/yxh/Documents/company/NeuralHDHair/female_halfbody_medium.obj")
    vertices_orig = deepcopy(body.vertices)
    lms = vertices_orig[[2636,2523,6052,2498,2481,3676,3760,1864,3640,3630,3557,3560,3457,3579,3326,3288,3068,4747,4724,4720,35,1066,4439,4482,4497,4399,4394],:]
    angs = [[0,0],[15,0],[-15,0],[0,30],[0,15],[0,-15],[0,-30]]
    angs_pair = readjson("angs_pair.json")
    for file in angs_pair:
        angs=angs_pair[file]
        for k in range(7,10):
            vertices = vertices_orig
            vertices = vertices+np.array([0.00703544,-1.58652416,-0.01121912])
            tform = transform.SimilarityTransform(rotation=[np.deg2rad(angs[k][0]),np.deg2rad(angs[k][1]),np.deg2rad(0)],dimensionality=3)#[0,30,0] 从上往下看顺时针旋转v3；[15,0,0] 向下旋转v1
            # print(tform.params)
            body.vertices = transform.matrix_transform(vertices, tform.params)+np.array([-0.00703544,1.58652416,0.01121912])
            render(body,bOrtho=True,preview_file=f"/media/yxh/My Passport/ths/neuraldata1/{file[:-5]}_v{k}/body.png")