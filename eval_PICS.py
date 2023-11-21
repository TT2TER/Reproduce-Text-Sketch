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

def get_cond_color(cond_image, mask_size=64):
    #improved form https://github.com/jinxixiang/color_controlnet
    #rectangular palette
    cond_image = to_pil_image(cond_image)
    cond_image = enhance_image(cond_image, 2.0, 1.1)
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.Resampling.BICUBIC)
    color = cond_image.resize((H, W), Image.Resampling.NEAREST)
    return color


def get_loss(args):
    if args.loss == 'clip':
        args_clip = Namespace()
        args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(x, xhat, clip_model, clip_preprocess, 'cuda:0')#計算兩個向量的余弦相似度
    else:
        sys.exit('Not a valid loss')

prompt_pos = 'high quality, high resolution, Leica camera effect, Leica photography'
prompt_neg = 'disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art, overexposure, color distortion'

def encode_rcc(model, clip, preprocess, ntc_sketch, im, N=5, i=0):
    """
    Generates canny map and caption of image. 
    Then uses ControlNet to generate codebook, and select minimum distortion index.
    Inputs: 
        model: ControlNet model
        clip: CLIP model
        im: image to compress
        N: number of candidates to generate
    Outputs:
        canny_map: np.array containing canny edge map
        caption: text string containing caption
        idx: index selected
        seed: random seed used
    """
    print("start encode_rcc")
#start of cannymap
    apply_canny = HEDdetector()
    canny_map = HWC3(apply_canny(im))

    # compress sketch
    sketch_canny = Image.fromarray(canny_map)
    sketch_canny = ntc_preprocess(sketch_canny).unsqueeze(0)#unsqueeze(0)表示在第0维增加一维
    with torch.no_grad():
        canny_dict = ntc_sketch.compress(sketch_canny)#压缩
        sketch_recon = ntc_sketch.decompress(canny_dict['strings'], canny_dict['shape'])['x_hat'][0]#解压缩
        sketch_recon = adjust_sharpness(sketch_recon, 2)#调整锐度
        sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))#处理后的图片
    sketch_recon = Image.fromarray(sketch_recon)
    #查看sketch_recon
    # sketch_recon.save(f'recon_examples/PICS_clip_ntclam1.0/CLIC2020_sketch/{i}_sketch_recon_mayutest.png')
#end of cannymap
#start of colormap
    #TODO: 处理colormap，然后压缩解压缩
    sketch_color = get_cond_color(im)
    with torch.no_grad():
        color_dict = ''
        pass
        #color_dict = ntc_sketch.compress(color_map)#TODO:这里应该换一个压缩模型
        #https://interdigitalinc.github.io/CompressAI/zoo.html
        #搞清楚这个仓库里的压缩模型哪来的
        #看起来好像是自己训练来的
        #print(color_dict)
        #TODO:解压缩
    color_recon = sketch_color
#end of colormap

    # Optionally load saved captions
    if i < 20:
        print('load saved captions')
        with open(f'recon_examples/PICS_clip_ntclam1.0/CLIC2020_recon/{i}_caption.yaml', 'r') as file:
            caption_dict = yaml.safe_load(file)
        caption = caption_dict['caption']
    else:
        print('optimize prompt')
        caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])#得到最优promt
    
    guidance_scale = 9
    num_inference_steps = 25
    control_images=[sketch_recon, color_recon]
    # improved from https://github.com/huggingface/diffusers/issues/5254
    # n_batches = N // 8 + 1
    # images = []
    # for b in range(n_batches):
    images = model(
        f'{caption}, {prompt_pos}',
        control_images,#模型的输入图像，为模拟压缩解压缩后的sketch
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N ,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        controlnet_conditioning_scale=[0.9, 1.0],
        ).images
    loss = loss_func([Image.fromarray(im)]*N, images).squeeze()
    idx = torch.argmin(loss)
    
    return caption, sketch_canny, canny_dict, sketch_color, color_dict, idx

def recon_rcc(model,  ntc_sketch, caption, canny_dict, color_dict, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    print("start recon_rcc")
    # # decode sketch
    with torch.no_grad():#解压缩草图
        sketch_canny = ntc_sketch.decompress(canny_dict['strings'], canny_dict['shape'])['x_hat'][0]
        sketch_canny = adjust_sharpness(sketch_canny, 2)
    sketch_canny = HWC3((255*sketch_canny.permute(1,2,0)).numpy().astype(np.uint8))
    sketch_canny = Image.fromarray(sketch_canny)

    sketch_color = color_dict#测试时color_dict暂时是sketch_color

    sketch= [sketch_canny, sketch_color]
    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        f'{caption}, {prompt_pos}',
        sketch,
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(8)],
        num_images_per_prompt=8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        controlnet_conditioning_scale=[0.9, 1.0],
        ).images

    return images[idx], sketch

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

    # Load ControlNet
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
    controlnet_hed = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16)#controlnet的參數
    controlnet_color = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-color-diffusers", torch_dtype=torch.float16)
    controlnet = [controlnet_hed, controlnet_color]
    model = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=controlnet, torch_dtype=torch.float16, revision="fp16",
    )
 
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)#https://huggingface.co/docs/diffusers/api/schedulers/unipc
    model.enable_xformers_memory_efficient_attention()
    model.enable_model_cpu_offload()#Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance.
    
    # Load loss
    loss_func = get_loss(args)

    # Load CLIP
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')


    # import json
    args_ntc = Namespace()
    args_ntc.model_name = 'Cheng2020AttentionFull'
    # args_ntc.model_name ='mbt2018_mean_Vimeo90k'
    args_ntc.lmbda = args.lam_sketch
    args_ntc.dist_name_model = "ms_ssim"
    args_ntc.orig_channels = 1
    ntc_sketch = models_compressai.get_models(args_ntc)
    saved = torch.load(f'models_ntc/OneShot_{args_ntc.model_name}_CLIC_HED_{args_ntc.dist_name_model}_lmbda{args_ntc.lmbda}.pt')
    # saved = torch.load(f'models_ntc/OneShot_{args_ntc.model_name}_{args_ntc.dist_name_model}_lmbda{args_ntc.lmbda}.pt')
    ntc_sketch.load_state_dict(saved)
    ntc_sketch.eval()#推理，评估模式
    ntc_sketch.update()#更新模型

    # Make savedir
    # save_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_recon'
    # sketch_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_sketch'
    #以下是测试用的地址
    save_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}_testcolor/{args.dataset}_recon'
    sketch_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}_testcolor/{args.dataset}_sketch'
    #以上是测试用的地址
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(sketch_dir).mkdir(parents=True, exist_ok=True)

    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        # Resize to 512
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        #permute:将tensor的维度换位，permute(1,2,0)表示将tensor的第一维换到第三维，第二维换到第一维，第三维换到第二维
        im = resize_image(HWC3(x_im), 512)
        # im = HWC3(x_im)
        
        # Encode and decode
        # caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N)
        # caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N, i)
        # xhat, sketch_recon = recon_rcc(model, ntc_sketch, caption, sketch_dict, idx,  args.N)

        #测试color用的encode和decode
        caption, sketch_canny, canny_dict, sketch_color, color_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N, i)
        xhat, sketch_recon = recon_rcc(model, ntc_sketch, caption, canny_dict, sketch_color, idx,  args.N)

        # Save ground-truth image
        im_orig = Image.fromarray(im)
        im_orig.save(f'{sketch_dir}/{i}_gt.png')

        # Save reconstructions
        xhat.save(f'{save_dir}/{i}_recon.png')

        # Save sketch images
        im_sketch_canny = to_pil_image(sketch_canny[0])
        im_sketch_canny.save(f'{sketch_dir}/{i}_sketch_canny.png')

        im_sketch_color = sketch_color
        im_sketch_color.save(f'{sketch_dir}/{i}_sketch_color.png')

        im_sketch_recon_canny = sketch_recon[0]
        im_sketch_recon_canny.save(f'{sketch_dir}/{i}_sketch_recon_canny.png')

        im_sketch_recon_color = sketch_recon[1]
        im_sketch_recon_color.save(f'{sketch_dir}/{i}_sketch_recon_color.png')

        # Compute rates
        bpp_sketch = sum([len(bin(int.from_bytes(s, sys.byteorder))) for s_batch in canny_dict['strings'] for s in s_batch]) / (im_orig.size[0]*im_orig.size[1])
        bpp_caption = sys.getsizeof(zlib.compress(caption.encode()))*8 / (im_orig.size[0]*im_orig.size[1])

        compressed = {'caption': caption,
                      'prior_strings':canny_dict['strings'][0][0],
                      'hyper_strings':canny_dict['strings'][1][0],
                      'bpp_sketch' : bpp_sketch,
                      'bpp_caption' : bpp_caption,
                      'bpp_total' : bpp_sketch + bpp_caption + math.log2(args.N) / (im_orig.size[0]*im_orig.size[1])
                      }
        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            file.write(caption)