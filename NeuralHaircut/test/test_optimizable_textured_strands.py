# 用训练得到的头发网络模型，附着在其他头皮上进行生长
from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
import yaml
import torch
import os
import trimesh
if __name__=="__main__":
    rFolder = os.path.dirname(__file__)
    conf_path = "./NeuralHaircut/configs/hair_strands_textured.yaml"
    device = "cuda"
    with open(conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', "person_0")
        config = yaml.load(replaced_conf, Loader=yaml.Loader)

    config['textured_strands']["path_to_mesh"]="/home/algo/yangxinhang/NeuralHDHair/NeuralHaircut/data/meishutoupi1.obj"#美术提供的base模型
    strands = OptimizableTexturedStrands(**config['textured_strands'], diffusion_cfg=config['diffusion_prior'],inference_on_other_scalp=True).to(device)
    strand_path = os.path.join(rFolder,'../implicit-hair-data/data/monocular/person_0/ckpt_114000.pth')
    # strand_path = "/home/algo/yangxinhang/NeuralHDHair/NeuralHaircut/exps_second_stage/second_stage_person_0/person_0/neural_strands_w_camera_fitted/2024-01-23_20:31:36/hair_primitives/ckpt_016000.pth"
    state_dict = torch.load(strand_path, map_location=device)
    del state_dict['strands']['local2world']
    del state_dict['strands']['origins']
    del state_dict['strands']['uvs']
    strands.load_state_dict(state_dict['strands'])
    strands,_,_=strands.forward_inference_guides(100)
    trimesh.PointCloud(strands.reshape(-1, 3).detach().cpu()).export('strands_test.ply')
    
    