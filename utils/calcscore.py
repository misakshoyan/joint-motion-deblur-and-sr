import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os
import argparse
import math
import cv2

import matplotlib.pyplot as plt

def psnr_my(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

# CNLRN metrics

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

 # end of CNLRN metrics



parser = argparse.ArgumentParser(description='calculates PSNR and SSIM')

parser.add_argument('--result', type=str, help='Result images')
parser.add_argument('--ground_truth', type=str, help='Ground truth images')

args = parser.parse_args()

res_dir = args.result
grth_dir = args.ground_truth

res_files = os.listdir(res_dir)
grth_files = os.listdir(grth_dir)

total_psnr = 0
# total_psnr_my = 0
total_ssim = 0

for im_name in tqdm(res_files):
    # print(im_name)
    # print(res_dir)
    # print(grth_dir)
    im1_path = os.path.join(res_dir, im_name)
    im2_path = os.path.join(grth_dir, im_name)
    # print(im1_path, im2_path)

    im1 = img_as_float(imread(im1_path))
    im2 = img_as_float(imread(im2_path))

    # plt.imshow(np.hstack([im1, im2]))
    # plt.show()

    # total_psnr += psnr(im1, im2)
    # total_psnr_my += psnr_my(im1, im2)
    # total_ssim += ssim(im1, im2, multichannel=True)
    # print(im1)
    total_psnr += calculate_psnr(im1*255, im2*255)
    total_ssim += ssim(im1*255, im2*255)

n = len(res_files)
print("Result for", res_dir.split('\\')[-1])
print(n)
print("Images: {}\nTotal PSNR: {}\nTotal SSIM: {}\n"
      .format(n,
              total_psnr / n,
              total_ssim / n))