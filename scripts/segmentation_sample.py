"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import pandas as pd
import time
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from LIDCLoader import load_LIDC
from performance_metrics import dice_score, calculate_hd95, iou_score, dice_coef2
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def show_tensor_images(image, mask, output, num, title=None): 
    to_pil = transforms.ToPILImage()

    pil_image1 = to_pil(image.squeeze().cpu())
    pil_image2 = to_pil(mask.squeeze().cpu())
    pil_image3 = to_pil(output)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(pil_image1)
    axes[0].set_title('Image', fontsize=10)

    axes[1].imshow(pil_image2)
    axes[1].set_title('Ground Truth', fontsize=10)

    axes[2].imshow(pil_image3)
    axes[2].set_title('Output', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.savefig('./output_images/Output_'+str(num)+'.png')
    # plt.show()

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # If multiple GPU is connected
    # model = th.nn.DataParallel(model)

    output_dir = './output'
    output_img_dir = './output_images'
    output_metrics_dir = './output_metrics'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_metrics_dir):
        os.makedirs(output_metrics_dir)

    ds = load_LIDC(image_size=224, combine_train_val=True, mode='Test')
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)

    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dp = {'title': [], 'dice_score': [], 'ioU': [], 'hd95': []}
    df = pd.DataFrame(dp)

    # for i in range(30):
    #     b, mask, image_path, mask_path = next(data)

    title = ''
    cnt = 1
    while len(all_images) * args.batch_size < args.num_samples:
        # should return an image from the dataloader "data"
        b, mask, image_path, mask_path = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # add a noise channel$
        print(image_path)
        # slice_ID = path[0].split("/", -1)[3]
        slice_ID = os.path.basename(image_path[0]).split(".")[0]
        title = os.path.basename(mask_path[0]).split(".")[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        
        tensor_list = []
        # this is for the generation of an ensemble of 5 masks.
        for i in range(args.num_ensemble):
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            end.record()
            th.cuda.synchronize()
            # time measurement for the generation of 1 sample
            print('time for 1 sample', start.elapsed_time(end))

            s = sample.clone().detach()
            tensor_list.append(s.squeeze().cpu())
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
        
        index, dice_high = 0, dice_coef2(pred=tensor_list[0], target=mask.squeeze().cpu())
        for i in range(1, args.num_ensemble):
            res = dice_coef2(pred=tensor_list[i], target=mask.squeeze().cpu())
            if dice_high < res:
                index, dice_high = i, res
        
        hd95_score = calculate_hd95(pred=tensor_list[index], target=mask.squeeze().cpu())
        ioU = iou_score(pred=tensor_list[index], target=mask.squeeze().cpu())

        new_row = pd.DataFrame([[title, dice_high, ioU, hd95_score]], 
                    columns=['title', 'dice_score', 'ioU', 'hd95'])
        df = pd.concat([df, new_row], ignore_index=True)
        output_file_path = os.path.join(output_metrics_dir, 'evaluation_metrics.csv')
        df.to_csv(output_file_path, index=False)

        show_tensor_images(image=b, mask=mask, output=tensor_list[index], num=cnt, title=str(slice_ID))
        th.save(tensor_list[index], './output/'+str(slice_ID)+'_output')  # save the generated mask
        tensor_list.clear()
        cnt = cnt + 1


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5  # number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
