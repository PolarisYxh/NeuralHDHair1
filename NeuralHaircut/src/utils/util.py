import torch
import numpy as np
import os
from glob import glob
import contextlib
import random
from skimage import transform
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras,FoVOrthographicCameras
def align_pyrender_camera(pos=None,intri=None,device='cpu'):
    '''
    param:
        pos:torch.tensor相机位姿矩阵(从image_filter.py得到self.cam_pose = x2@four_by_four@x1@np.linalg.inv(m[2]));shape:[1,4,4]
        intri:torch.tensor 相机内参矩阵，shape:[1,4,4]
    '''
    # pytorch3d相机对齐pyrender相机得到RT，法一，使用相机位姿矩阵(从image_filter.py得到self.cam_pose = x2@four_by_four@x1@np.linalg.inv(m[2]))
    if not isinstance(pos,torch.Tensor):
        pos = torch.tensor([[ 9.61313906e-01,  7.01898160e-02, -2.66362467e-01, -7.95505687e-02],
                        [ 6.93889390e-18,  9.66990113e-01,  2.54813897e-01, 1.65589804e+00],
                        [ 2.75455212e-01, -2.44956142e-01,  9.29581042e-01, 2.64301285e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])[None]# for test
    
    cam_T=pos[:,:3,3].to(torch.float32)
    R = pos[:,:3,:3]
    rot_matrix = torch.from_numpy(transform.SimilarityTransform(rotation=[0,np.deg2rad(180),0],dimensionality=3).params)[:3,:3].to(torch.float32).to('cuda')
    R = (R@rot_matrix).to(torch.float32)#绕y轴旋转180度，即第一第三列取反
    T = -torch.bmm(R.transpose(1, 2), cam_T[:, :, None])[:, :, 0]
    # pytorch3d相机对齐pyrender相机得到RT，法二，使用目标位置和相机位置T
    # pos1 = np.array([[ 9.61313906e-01,  7.01898160e-02, -2.66362467e-01, -7.95505687e-02],
    #                 [ 6.93889390e-18,  9.66990113e-01,  2.54813897e-01, 1.65589804e+00],
    #                 [ 2.75455212e-01, -2.44956142e-01,  9.29581042e-01, 2.64301285e-01],
    #                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    # tar_pose = np.array([0, 1.58652416, 0])#NeuralHDHair默认的
    # tar_pose = torch.from_numpy(tar_pose)[None].to(torch.float32)
    # cam_T=torch.from_numpy(pos1[:3,3])[None].to(torch.float32)
    # R0, T0 = look_at_view_transform(eye=torch.from_numpy(pos1[:3,3])[None].to(torch.float32),at=tar_pose)
    # look_at_view_transform
    # R1 = look_at_rotation(cam_T, tar_pose) ==R0
    # T1 = -torch.bmm(R1.transpose(1, 2), cam_T[:, :, None])[:, :, 0] ==T0
    
    # camera intrinsic
    if not isinstance(intri,torch.Tensor):
        cam = 0.28347224#96*0.00567194(voxel_size)-0.261034
        far = 0.261034
        xymag = 0.36300416
        cams = [FoVOrthographicCameras(device=device,znear=0.0001,zfar=cam+far,R=R,T=T,max_x=xymag,min_x=-xymag,max_y=xymag,min_y=-xymag)]
    else:
        intri = intri.to(torch.float32).to(device)
        cams = [FoVOrthographicCameras(device=device,R=R,T=T,K=intri)]
    return cams
def scale_mat(mat, scale_factor):
    mat[0, 0] /= scale_factor
    mat[1, 1] /= scale_factor
    mat[0, 2] /= scale_factor
    mat[1, 2] /= scale_factor
    return mat

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded. B x C x ...
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    for submodule in module.children():
        param_to_buffer(submodule)
    for name, param in dict(module.named_parameters(recurse=False)).items():
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param, persistent=False)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    return image.astype(np.uint8).copy()

def glob_imgs(path):
    imgs = []
    for ext in ['*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG', '*.npy', '*.NPY']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def fill_tensor(x, mask, c=0):
    if x is not None:
        out = x.new_ones((mask.shape[0], *x.shape[1:])) * c
        out[mask] = x
        return out 
    else:
        return x
    
@contextlib.contextmanager
def freeze_gradients(model):
    is_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    yield
    if is_training:
        model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    