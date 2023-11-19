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
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness
import yaml
import sys, zlib
from argparse import ArgumentParser, Namespace
# import lpips

def get_loss(args):
    if args.loss == 'clip':
        args_clip = Namespace()
        args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(x, xhat, clip_model, clip_preprocess, 'cuda:0')
    else:
        sys.exit('Not a valid loss')

prompt_pos = 'high quality'
prompt_neg = 'disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art'

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
    apply_canny = HEDdetector()
    canny_map = HWC3(apply_canny(im))

    # compress sketch
    sketch = Image.fromarray(canny_map)
    sketch = ntc_preprocess(sketch).unsqueeze(0)
    with torch.no_grad():
        sketch_dict = ntc_sketch.compress(sketch)
        sketch_recon = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
        sketch_recon = adjust_sharpness(sketch_recon, 2)
        sketch_recon = HWC3((255*sketch_recon.permute(1,2,0)).numpy().astype(np.uint8))
    
    # Optionally load saved captions
    # if i > 0:
    with open(f'recon_examples/PICS_clip_ntclam1.0/CLIC2020_recon/{i}_caption.yaml', 'r') as file:
        caption_dict = yaml.safe_load(file)
    caption = caption_dict['caption']
    # else:
    #     caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])
    
    guidance_scale = 9
    num_inference_steps = 25

    # n_batches = N // 8 + 1
    # images = []
    # for b in range(n_batches):
    images = model(
        f'{caption}, {prompt_pos}',
        Image.fromarray(sketch_recon),
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N ,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        ).images
    loss = loss_func([Image.fromarray(im)]*N, images).squeeze()
    idx = torch.argmin(loss)
    
    return caption, sketch, sketch_dict, idx

def recon_rcc(model,  ntc_sketch, caption, sketch_dict, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    # decode sketch
    with torch.no_grad():
        sketch = ntc_sketch.decompress(sketch_dict['strings'], sketch_dict['shape'])['x_hat'][0]
        sketch = adjust_sharpness(sketch, 2)
    sketch = HWC3((255*sketch.permute(1,2,0)).numpy().astype(np.uint8))

    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    images = model(
        f'{caption}, {prompt_pos}',
        Image.fromarray(sketch),
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(8)],
        num_images_per_prompt=8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        ).images

    return images[idx], sketch

def ntc_preprocess(image):
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
    parser.add_argument('--lam_sketch', default=1.0, type=str)

    args = parser.parse_args()
    # dm = Kodak(root='~/data/Kodak', batch_size=1)
    dm = dataloaders.get_dataloader(args)

    # Load ControlNet
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=controlnet, torch_dtype=torch.float16, revision="fp16",
    )
    print("here")
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_xformers_memory_efficient_attention()
    model.enable_model_cpu_offload()
    
    # Load loss
    loss_func = get_loss(args)

    # Load CLIP
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')

    # from argparse import Namespace
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
    ntc_sketch.eval()
    ntc_sketch.update()

    # Make savedir
    save_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_recon'
    sketch_dir = f'recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_sketch'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(sketch_dir).mkdir(parents=True, exist_ok=True)

    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        # Resize to 512
        x = x[0]
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        # im = HWC3(x_im)
        
        # Encode and decode
        # caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N)
        caption, sketch, sketch_dict, idx = encode_rcc(model, clip, clip_preprocess, ntc_sketch, im, args.N, i)
        xhat, sketch_recon = recon_rcc(model, ntc_sketch, caption, sketch_dict, idx,  args.N)

        # Save ground-truth image
        im_orig = Image.fromarray(im)
        im_orig.save(f'{sketch_dir}/{i}_gt.png')

        # Save reconstructions
        xhat.save(f'{save_dir}/{i}_recon.png')

        # Save sketch images
        im_sketch = to_pil_image(sketch[0])
        im_sketch.save(f'{sketch_dir}/{i}_sketch.png')

        im_sketch_recon = Image.fromarray(sketch_recon)
        im_sketch_recon.save(f'{sketch_dir}/{i}_sketch_recon.png')

        # Compute rates
        bpp_sketch = sum([len(bin(int.from_bytes(s, sys.byteorder))) for s_batch in sketch_dict['strings'] for s in s_batch]) / (im_orig.size[0]*im_orig.size[1])
        bpp_caption = sys.getsizeof(zlib.compress(caption.encode()))*8 / (im_orig.size[0]*im_orig.size[1])

        compressed = {'caption': caption,
                      'prior_strings':sketch_dict['strings'][0][0],
                      'hyper_strings':sketch_dict['strings'][1][0],
                      'bpp_sketch' : bpp_sketch,
                      'bpp_caption' : bpp_caption,
                      'bpp_total' : bpp_sketch + bpp_caption + math.log2(args.N) / (im_orig.size[0]*im_orig.size[1])
                      }
        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            # file.write(caption)