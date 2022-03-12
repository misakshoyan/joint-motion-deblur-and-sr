"""
## Single Image Joint Motion Deblurring and Super-Resolution
## Using the Multi-Scale Channel Attention Modules
## Misak Shoyan
##
##
## Based on 'Multi-Stage Progressive Image Restoration'
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import math

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data, get_test_data_deblur
from local_arch import MPRNetLocal
# from MPRNet_SR import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

from torchvision.utils import make_grid

# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# self-ensemble strategy as in CNLRN
def flipx8_forward(model, inp):
    """Flip testing with X8 self ensemble
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """

    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).type_as(v)
        return ret

    lr_list = [inp]
    sr_list = []
    for tf in 'v', 'h', 't':
        lr_list.extend([_transform(t, tf) for t in lr_list])

    with torch.no_grad():
        for aug in lr_list:
            # cnt = cnt+1
            dbs, sr = model(aug)
            sr_list.append(sr)

    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)

    output = output.data.float().cpu()
    return output

# Used to calculate the PSNR/SSIM metrics as in CNLRN for fair comparison.
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)




parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--flip_test', action='store_true', help='using self ensemble if true')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNetLocal()
# model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
print("Flip test: ", args.flip_test)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

rgb_dir_test = args.input_dir
print(rgb_dir_test)
# test_dataset = get_test_data_deblur(rgb_dir_test)
test_dataset = get_test_data(rgb_dir_test)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

result_dir  = args.result_dir
utils.mkdir(result_dir)

psnr_all = 0
ssim_all = 0
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        target = data_test[1].cuda()

        # starter.record()

        if args.flip_test:
            restored = flipx8_forward(model_restoration, input_)
        else:
            restored_dbs, restored = model_restoration(input_)

        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print("ellapsed time on image: ", curr_time)

        restored2 = tensor2img(restored.squeeze(0))
        target2 = tensor2img(target.squeeze(0))

        psnr_score = utils.calculate_psnr(restored2, target2)
        psnr_all += psnr_score

        ssim_score = utils.ssim(restored2, target2)
        ssim_all += ssim_score
        print("PSNR/SSIM_{} = {:.4f}/{:.4f}".format(ii, psnr_score, ssim_score))

        # print("restored1 PSNR: ", psnr_score)

        #####
        restored = torch.clamp(restored,0,1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, str(ii) + '.png')), restored_img)

print("Total PSNR: ", psnr_all/300)
print("Total SSIM: ", ssim_all/300)