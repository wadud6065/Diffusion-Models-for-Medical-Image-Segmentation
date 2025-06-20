# Diffusion Models for Medical Image Segmentation

This repository is inspired by the PyTorch implementation of OpenAI's [improved-diffusion](https://github.com/openai/improved-diffusion) and the research paper [Diffusion Models for Implicit Image Segmentation Ensembles](https://arxiv.org/abs/2112.03145), [GitHub Repo](https://github.com/JuliaWolleb/Diffusion-based-Segmentation) by Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe Valmaggia, and Philippe C. Cattin.

## Data

The data is converted into ".png" file format which can be found in the directory _./data_image_. It follows the following structure:

```
data_image
└───noise
│   └───1_IM-0001-0059_anon.png
│   └───2_IM-0383-0001_anon.png
│   └───3_IM-0359-0014_anon.png
│   └───5_IM-0040-0020_anon.png
│   └───6_IM-0148-0007_anon.png
│       │  ...
└───clean
│   └───1_IM-0001-0059_anon.png
│   └───2_IM-0383-0001_anon.png
│   └───3_IM-0359-0014_anon.png
│   └───5_IM-0040-0020_anon.png
│   └───6_IM-0148-0007_anon.png
│       │  ...

```

clean folder contains the clean images of each noisy images.

## Usage

We set the flags as follows:

```
MODEL_FLAGS="--num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
```

To train the denoising model, run

```
python scripts/denoising_train.py --data_dir ./data/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```

The model will be saved in the _results_ folder.
For sampling an ensemble of 5 segmentation masks with the DDPM approach, run:

```
python scripts/denoising_sample.py  --data_dir ./data/testing  --model_path ./results/savedmodel.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
```

The generated segmentation masks will be stored in the _results_ folder.
