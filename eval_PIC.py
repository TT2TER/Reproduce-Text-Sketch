import torch
from dataloaders import CLIC, Kodak
import matplotlib.pyplot as plt
import numpy as np
import math
from annotator.hed import HEDdetector
from annotator.util import HWC3, resize_image
import tqdm
import pathlib
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import prompt_inversion.optim_utils as prompt_inv
import prompt_inversion.open_clip as open_clip 
import dataloaders
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_pil_image, adjust_sharpness, to_tensor
import yaml
import sys, zlib
from argparse import ArgumentParser, Namespace
# import lpips
# 測試github推送

def get_loss(args):
    # 這段代碼根據輸入參數args.loss選擇不同的loss函數，返回一個函數
    # if args.loss == 'lpips':
    #     return lpips.LPIPS(net='alex') 
    if args.loss == 'clip':
        args_clip = Namespace()#Namespace是一個簡單的類，用於創建具有屬性的對象，屬性可以通過點表示法訪問
        args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))#將json文件中的內容更新到args_clip中
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')
        #創建CLIP模型
        print("loss: clip")
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(x, xhat, clip_model, clip_preprocess, 'cuda:0')#計算兩個向量的余弦相似度
    else:
        sys.exit('Not a valid loss')


prompt_pos = 'high quality'
prompt_neg = 'disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art'

def encode_rcc(model, clip, preprocess, im, N=5, i=0):
    """
    Generates canny map and caption of image.  這裡只用了caption信息，沒有用到canny_map
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
    # Optionally load saved captions (for consistency)
    # if i > 0:
    #     with open(f'recon_examples/SD_pi+hed_lpips_sketch0.5/DIV2K_recon/{i}_caption.yaml', 'r') as file:
    #         caption_dict = yaml.safe_load(file)
    #     caption = caption_dict['caption']
    # else:
    caption = prompt_inv.optimize_prompt(clip, preprocess, args_clip, 'cuda:0', target_images=[Image.fromarray(im)])
    #調用了prompt_inversion中的optimize_prompt函數，目的是優化prompt
    #參數包括：clip模型，clip_preprocess與訓練模型，args_clip，設備，目標圖片
    #caption是一個字符串，是優化後的prompt
    guidance_scale = 9 #指導力度
    num_inference_steps = 25 #推斷步數

    images = model(
        f'{caption}, {prompt_pos}', #拼接字符串，caption是優化後的prompt，prompt_pos是一個字符串，代表高質量
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],#生成器，生成N個隨機數
        num_images_per_prompt=N,#每個prompt生成N個圖片
        guidance_scale=guidance_scale,#指導力度
        num_inference_steps=num_inference_steps,#推斷步數
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,#負向prompt
        ).images

    loss = loss_func([Image.fromarray(im)]*N, images).squeeze()#計算優化後的prompt對應的圖片與原圖的loss
    idx = torch.argmin(loss)#TODO：這是啥？效果最好的圖片的索引？
    
    return caption, idx

def recon_rcc(model,  prompt, idx, N=5):
    """
    Takes canny map and caption to generate codebook. 
    Outputs codebook[idx], where idx is selected from encoder.
    Inputs:

    """
    # decode image
    guidance_scale = 9
    num_inference_steps = 25

    # n_batches = N // 8 + 1
    # images = []
    # for b in range(n_batches):
    images = model(
        f'{prompt}, {prompt_pos}',
        generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
        ).images
    #要保證encoder和decoder用來生成圖片的模型和seed等一致，才能保證生成的圖片是對應的
    return images[idx]

def ntc_preprocess(image):
    transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    image = transform(image)
    return image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--N', default=4, type=int)
    parser.add_argument('--dataset', default='CLIC2020', type=str)
    parser.add_argument('--data_root', default='/root/autodl-tmp/image_datasets', type=str)
    parser.add_argument('--loss', default='clip', type=str)

    args = parser.parse_args()
    dm = dataloaders.get_dataloader(args)#加載數據集

    # Load Stable Diffusion
    model_id = "stabilityai/stable-diffusion-2-1-base"#通過model_id找到模型，從huggingface上下載
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")#從huggingface上下載scheduler，並且加載(https://huggingface.co/docs/diffusers/api/schedulers/overview)

    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,#torch.float16是半精度浮點數，節省內存
        revision="fp16",
        )
    model = model.to('cuda:0')
    model.enable_xformers_memory_efficient_attention()#啟用xfomer的內存效率注意力，更有效利用內存
    # model.enable_attention_slicing()

    # Load loss
    loss_func = get_loss(args)

    # Make savedir,用來存放推斷結果
    save_dir = f'recon_examples/PIC_{args.loss}/{args.dataset}_recon'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load CLIP
    args_clip = Namespace()
    args_clip.__dict__.update(prompt_inv.read_json("prompt_inversion/sample_config.json"))
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(args_clip.clip_model, pretrained=args_clip.clip_pretrain, device='cuda:0')

    for i, x in tqdm.tqdm(enumerate(dm.test_dset), total=len(dm.test_dset)):#dm.test_dset是測試集，隨機選取一張圖片
        x = x[0]#x是一個tuple，x[0]是圖片，x[1]是標籤
        x_im = (255*x.permute(1,2,0)).numpy().astype(np.uint8)#將tensor轉換為numpy數組，並且轉換為uint8類型
        im = resize_image(HWC3(x_im), 512)#將圖片resize為512*512
        
        # caption, idx = encode_rcc(model, clip, clip_preprocess, im, args.N)
        caption, idx = encode_rcc(model, clip, clip_preprocess, im, args.N, i)#效果最好的prompt和效果最好的圖片的索引
        xhat = recon_rcc(model, caption, idx,  args.N)#返回的是效果最好的图片

        im_orig = Image.fromarray(im)#將numpy數組轉換為PIL圖片
        im_orig.save(f'{save_dir}/{i}_gt.png')#保存原圖

        # for j, im_recon in enumerate(xhat):
        #     im_recon.save(f'{save_dir}/{i}_recon_{j}.png')
        # im_recon = Image.fromarray(xhat)
        xhat.save(f'{save_dir}/{i}_recon.png')#保存重建圖片

        # im_sketch = Image.fromarray(sketch)
        # im_sketch = to_pil_image(sketch[0])
        # im_sketch.save(f'{save_dir}/{i}_sketch.png')

        # im_sketch_recon = Image.fromarray(sketch_recon)
        # im_sketch_recon.save(f'{save_dir}/{i}_sketch_recon.png')

        # Compute rates
        bpp_caption = sys.getsizeof(zlib.compress(caption.encode()))*8 / (im_orig.size[0]*im_orig.size[1])
        #計算文本提示的壓縮率，bits per pixel，即每個像素的比特數
        compressed = {'caption': caption,
                      'bpp_caption' : bpp_caption,
                      'bpp_total' : bpp_caption +  math.log2(args.N) / (im_orig.size[0]*im_orig.size[1])
                      }

        with open(f'{save_dir}/{i}_caption.yaml', 'w') as file:
            yaml.dump(compressed, file)
            # file.write(caption)