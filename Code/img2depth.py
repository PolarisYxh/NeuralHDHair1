import torch
import numpy as np
import imageio
from torch.autograd import Variable
import os
from tqdm import tqdm

from Models.hourglass import Model

import matplotlib.pyplot as plt
import cv2
class hairDepth:
    def __init__(self,model_path) -> None:
        self.model = Model().cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    def img2depth(self,rgb_img,mask):
        # rgb_img = imageio.imread(os.path.join(resized_path, item))[:,:, 0:3] / 255.
        # mask = (imageio.imread(os.path.join(output_seg_path, item))/255.>0.5)[:,:,0]
        # rgb_img = rgb_img
        rgb_img = Variable(torch.from_numpy(rgb_img).permute(2, 0, 1).float().unsqueeze(0)).cuda()

        depth_pred = self.model(rgb_img)
        depth_pred = depth_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        depth_pred_masked = depth_pred[:, :, 0] * mask - (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred)))
        max_val = np.nanmax(depth_pred_masked)
        min_val = np.nanmin(depth_pred_masked + 2 * (1 - mask) * (np.abs(np.nanmax(depth_pred)) + np.abs(np.nanmin(depth_pred))))
        depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val)*mask
        depth_pred_norm = np.clip(depth_pred_norm, 0., 1.)
        # cv2.imshow("1",(depth_pred_norm*255).astype('uint8'))
        # cv2.waitKey()
        # np.save(os.path.join(output_depth_path, item[:-3]+'npy'), depth_pred_norm)
        return depth_pred_norm