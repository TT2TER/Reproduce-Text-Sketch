import torch
from dataloaders import CLIC, Kodak
import matplotlib.pyplot as plt
import numpy as np
import math
from annotator.hed import HEDdetector
from annotator.util import HWC3, resize_image
import tqdm
import pathlib
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import models_compressai
import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 
import dataloaders
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness
import yaml
import sys, zlib
from argparse import ArgumentParser, Namespace
# import lpips

def enhance_image(image, contrast_factor, saturation_factor):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)
    return image

def get_cond_color(cond_image, mask_size=128):
    #improved form https://github.com/jinxixiang/color_controlnet
    #rectangular palette
    cond_image = to_pil_image(cond_image)
    cond_image = enhance_image(cond_image, 1.5, 1.2)
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.Resampling.BICUBIC)
    color = cond_image.resize((H, W), Image.Resampling.NEAREST)
    return color

def ntc_preprocess(image):
    #输入图像
    #将图像转化为灰度，再转化为tensor
    #返回图像
    transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()]
        )
    image = transform(image)
    return image

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--N', default=4, type=int)
    parser.add_argument('--dataset', default='CLIC2020', type=str)
    parser.add_argument('--data_root', default='/root/autodl-tmp/image_datasets', type=str)
    parser.add_argument('--loss', default='clip', type=str)
    parser.add_argument('--lam_sketch', default=1.0, type=str)#ntc_sketch的參數

    args = parser.parse_args()
    # dm = Kodak(root='~/data/Kodak', batch_size=1)
    dm = dataloaders.get_dataloader(args)

    sketch_dir = f'recon_examples/traindata/sketch'
    pathlib.Path(sketch_dir).mkdir(parents=True, exist_ok=True)

    for i, x in tqdm.tqdm(enumerate(dm.train_dset)):
        # Resize to 512
        print(i,'/',len(dm.train_dset))
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        #permute:将tensor的维度换位，permute(1,2,0)表示将tensor的第一维换到第三维，第二维换到第一维，第三维换到第二维
        im = resize_image(HWC3(x_im), 512)
        # im = HWC3(x_im)
        
        # Encode and decode
        # caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N)
        # caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N, i)
        # xhat, sketch_recon = recon_rcc(model, ntc_sketch, caption, sketch_dict, idx,  args.N)
        sketch_color = get_cond_color(im)
        

        im_sketch_color = sketch_color
        im_sketch_color.save(f'{sketch_dir}/{i}_sketch_color.png')