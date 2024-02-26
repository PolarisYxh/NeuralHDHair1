import numpy as np
import dataload.pyrender as pyrender
from skimage import transform
import cv2
import os
import sys
import random
sys.path.append(os.path.dirname(__file__))
import platform
plat = platform.system().lower()
if plat != 'windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
def render_strand(strands,segments,mesh=None,inference=True,width=256,vertex_colors=np.array([0, 0, 0, 255]),orientation=None,mask=False,intensity=3.0, strand_color = None, offscreen = True,cam_pos=[],matrix=[]):
    """_summary_

    Args:
        strands (_type_): _description_
        segments (_type_): _description_
        mesh (_type_, optional): _description_. Defaults to None.
        width (int, optional): _description_. Defaults to 256.
        vertex_colors (_type_, optional): _description_. Defaults to np.array([0, 0, 0, 255]).
        orientation (_type_, optional): _description_. Defaults to None.
        mask (bool, optional): _description_. Defaults to False.
        intensity (float, optional): _description_. Defaults to 3.0.
        strand_color (_type_, optional): _description_. Defaults to None.
        offscreen (bool, optional): _description_. Defaults to True.
        cam_pos (list, optional): _description_. Defaults to [].
        matrix (list, optional): _description_. Defaults to [].

    Returns:
        color: 方向图，R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
    """    
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1],bg_color=[0,0,0])
    if mesh:
        try:
            mesh.visual = mesh.visual.to_color()
        except:
            pass
        mesh.visual.vertex_colors = vertex_colors
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    # tri_geometry = mesh.geometry
    # for i,t in enumerate(tri_geometry):
    #     mesh = tri_geometry[t]
    #     mesh.visual = mesh.visual.to_color()
    #     mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
    #     mesh = pyrender.Mesh.from_trimesh(tri_geometry[t])
    #     scene.add(mesh)
    if mask:
        flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    else:
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=int(width / 1.0), point_size=1.0)
    lines = []
    # root = []
    start = 0
    for i,length in enumerate(segments):
        if length==1:
            continue
        if isinstance(orientation,np.ndarray):
            line = pyrender.Primitive(strands[start:start+length],color_0=orientation[i],mode=3)
        elif isinstance(strand_color,np.ndarray):
            line = pyrender.Primitive(strands[start:start+length],color_0=strand_color[start:start+length],mode=3)
        else:
            colors = [[255,0,0,255],[0,255,0,255],[0,0,255,255]]
            line = pyrender.Primitive(strands[start:start+length],color_0=random.choice(colors),mode=3)
        lines.append(line)
        start+=length
        # root.append(strand[0])
        
    m_line = pyrender.Mesh(lines, name="strands", is_visible=True)
    node = scene.add(m_line)
    
    aspectRatio = 1.0
    bOrtho = True
    if bOrtho:
        near = 0.0001
        xymag = 0.36300416
        cam = 0.28347224#96*0.00567194(voxel_size)-0.261034
        far = 0.261034
        pc = pyrender.OrthographicCamera(xymag,xymag,znear=near,zfar=cam+far)
        # pc = pyrender.OrthographicCamera(xymag, xymag, zfar=200)
    else:
        near = 2.5
        cam = 0.28347224+near
        far = 0.261034
        pc = pyrender.PerspectiveCamera(yfov=np.deg2rad(15),znear=near,zfar=cam+far,aspectRatio=aspectRatio)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = camera_pose = transform.SimilarityTransform(translation=(0, 1.6, -5), rotation=(np.deg2rad(180),np.deg2rad(0),0), dimensionality=3)
    if not mask:
        scene.add(light, light_pose)
        light1 = pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=intensity)
        light_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(light1, light_pose)
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.600],
        [0.0, 0.0, 1.0, 5.00],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # 1.58652416=64*0.00567194(voxel_size)+1.22352,-0.00703244=64*0.00567194(voxel_size)-0.3700396
    camera_pose = transform.SimilarityTransform(translation=(-0.00703244, 1.58652416, cam), rotation=(np.deg2rad(0),0,0), dimensionality=3)
    scene.add(pc, name="main_cam", pose=camera_pose.params)
    if not offscreen:
        # scene.main_camera_node.matrix = np.dot(move, camera_pose)
        pyrender.Viewer(scene)
    else:
        colors = []
        color, depth = r.render(scene, flags=flags)#depth:返回到相机真实的距离；pyrender renderer.py line 1156 change depth:_read_main_framebuffer，
        colors.append(color)
        # cv2.imshow("1",color)
        # cv2.waitKey()
        for i, move in enumerate(cam_pos):
            scene.main_camera_node.matrix = move
            color, depth = r.render(scene, flags=flags)
            colors.append(color)
            # cv2.imshow("1",color)
            # cv2.waitKey()
    matrix.append(np.dot(pc.get_projection_matrix(), np.linalg.inv(scene.main_camera_node.matrix)))
    if not inference:
        matrix.append(pc.get_projection_matrix())
        # print(pc.get_projection_matrix())
        matrix.append(np.linalg.inv(scene.main_camera_node.matrix))
    r.delete()
    inds = (depth != 0)
    depth[inds] = (depth[inds]-near)/(cam+far-near)#(96*0.00567194)
    # cv2.imwrite("2.png",(depth*255).astype('uint8'))
    depth1 = depth.copy()

    if inference:
        # 推理时  用奇胜透视投影相机得到的内外参,表示奇胜默认的源照片拍摄的相机内外参
        near=2.5
        cam = 0.28347224+near
        far = 0.261034
        pc = pyrender.PerspectiveCamera(yfov=np.deg2rad(15),znear=near,zfar=cam+far,aspectRatio=aspectRatio)
        #相机标准位置还是使用我这边计算得到的，人脸位姿之后可以使用奇胜优化得到的位姿数据
        camera_pose = transform.SimilarityTransform(translation=(-0.00703244, 1.58652416, cam), rotation=(np.deg2rad(0),0,0), dimensionality=3).params
        # matrix.append(np.dot(pc.get_projection_matrix(), np.linalg.inv(camera_pose)))
        matrix.append(pc.get_projection_matrix())
        matrix.append(np.linalg.inv(camera_pose))
    return depth, depth1,color
def render_cartoon(hair_mesh,mesh=None,width=256,hair_colors = np.array([0, 0, 0, 255]),mesh_colors=np.array([255, 0, 0, 255]),orientation=None,mask=False,intensity=3.0, strand_color = None, offscreen = True,cam_pos=[],matrix=[]):
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1],bg_color=[255,255,255])
    try:
        hair_mesh.visual = hair_mesh.visual.to_color()
    except:
        pass
    hair_mesh.visual.vertex_colors = hair_colors
    hair_mesh = pyrender.Mesh.from_trimesh(hair_mesh)
    scene.add(hair_mesh)
    if mesh:
        try:
            mesh.visual = mesh.visual.to_color()
        except:
            pass
        mesh.visual.vertex_colors = mesh_colors
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    # tri_geometry = mesh.geometry
    # for i,t in enumerate(tri_geometry):
    #     mesh = tri_geometry[t]
    #     mesh.visual = mesh.visual.to_color()
    #     mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
    #     mesh = pyrender.Mesh.from_trimesh(tri_geometry[t])
    #     scene.add(mesh)
    if mask:
        flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    else:
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=int(width / 1.0), point_size=1.0)
    
    aspectRatio = 1.0
    bOrtho = True
    if bOrtho:
        xymag = 0.36300416
        cam = 0.28347224#96*0.00567194(voxel_size)-0.261034
        far = 0.261034
        pc = pyrender.OrthographicCamera(xymag,xymag,znear=0.0001,zfar=cam+far)
        # pc = pyrender.OrthographicCamera(xymag, xymag, zfar=200)
    else:
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 15, aspectRatio=aspectRatio)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    if not mask:
        scene.add(light, light_pose)
        light1 = pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=intensity)
        light_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(light1, light_pose)
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.600],
        [0.0, 0.0, 1.0, 5.00],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # 1.58652416=64*0.00567194(voxel_size)+1.22352,-0.00703244=64*0.00567194(voxel_size)-0.3700396
    camera_pose = transform.SimilarityTransform(translation=(-0.00703244, 1.58652416, cam), rotation=(np.deg2rad(0),0,0), dimensionality=3)
    scene.add(pc, name="main_cam", pose=camera_pose.params)
    if not offscreen:
        # scene.main_camera_node.matrix = np.dot(move, camera_pose)
        pyrender.Viewer(scene)
    else:
        colors = []
        color, depth = r.render(scene, flags=flags)
        colors.append(color)
        # cv2.imshow("1",color)
        # cv2.waitKey()
        for i, move in enumerate(cam_pos):
            scene.main_camera_node.matrix = move
            color, depth = r.render(scene, flags=flags)
            colors.append(color)
            # cv2.imshow("1",color)
            # cv2.waitKey()
    matrix.append(np.dot(pc.get_projection_matrix(), np.linalg.inv(scene.main_camera_node.matrix)))
    r.delete()
    depth = depth/(96*0.00567194)
    # depth = depth.astype('uint8')
    depth1 = depth.copy()
    # depth[np.where(color[:,:,0]==color[:,:,1])]=0
    # y = np.max(depth)
    # depth[depth==0]=255
    # x=np.min(depth)#49/255*95=18.2
    # min_z = np.min(depth[:,:])
    # cv2.imshow("1",color)
    # cv2.waitKey()
    return depth, depth1,color