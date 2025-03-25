# [CVPR2025] Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models
This repository hosts the PyTorch implementation for our paper "Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models", accepted at CVPR 2025.

![Semantic watermark forgery. The attacker can transfer the watermark from a watermarked reference image requested by Alice (here: the diving cat) into any cover image (here: the moon landing). The obtained image will be detected as watermarked and attributed to Alice by the service provider, eroding the trust in watermark-based detection and attribution of AI-generated content.](media/frontpage.jpg)

[![arXiv](https://img.shields.io/badge/arXiv-2412.03283-b31b1b.svg)](https://arxiv.org/abs/2412.03283)


Please cite as follows:
```
@InProceedings{Mueller_2025_CVPR,
  author    = {Andreas Müller and Denis Lukovnikov and Jonas Thietke and Asja Fischer and Erwin Quiring},
  title     = {Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
}
```

This repository builds upon and incorporates code from the following repositories:
1. https://github.com/YuxinWenRick/tree-ring-watermark - Please adhere to their license.
2. https://github.com/lthero-big/A-watermark-for-Diffusion-Models - Please adhere to their license.

Please see the [LICENSE](./LICENSE) for this repository.

<br><br>

## Setup Instructions

We tested this setup on ubuntu 20.04.6 LTS with a NVIDIA A40 GPU with 45GB of VRAM. The attacks should work for smaller GPUs, depending on the model. Please note that in order to use FLUX.1 as the target model, a GPU with support for bfloat might be needed.

### 1. Install Conda
Make sure you have Conda installed on your system. You can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

### 2. Create a Conda Environment
Create a new Conda environment Python 3.10. Run the following command in your terminal:

```bash
conda create --name semantic-forgery python=3.10
conda activate semantic-forgery
```

Install packages
```bash
pip install -r requirements.txt
```

<br><br>

## Attacks on Semantic Watermarks
We introduce attacks on semantic watermarks for diffusion models, that is watermarks that are embedded by preparing the intitial latent input to produce watermarked images, which are later verified by using inversion.

The two semantic watermarking schemes we evaluate are Tree-Ring (TR) and Gaussian Shading (GS).

There are three attacks.

The first two are similar as they use the same optimization mechanism we call *Imprinting*, and hence refer to them as *Imprinting* attacks:
- *Imprint-Forgery* (*Imprint-F* for short) attack (for forging the semantic "imprint" from a watermarked reference image and apply it on an arbitrary cover image)
- *Imprint-Removal* (*Imprint-R* for short) attack (for removing the semantic "imprint" from a watermarked reference image)

The third attack does not use the optimization procedure, but just inverts and regenerates a given reference image with a totally different, possibly harmful prompt.
We refer to this attack as:
- *Reprompting* ("Reprompt" for short)

If this method is used with multiple attacker prompts on a single reference image (and also multiple runs of resampling the zT bins in the case of GS), you get the *Reprompt+* attack from the paper.

To avoid downloading large datasets unnecessarily, we provide a small sample of prompts from the SD-Prompts (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts) and the I2P (https://huggingface.co/datasets/AIML-TUDA/i2p) dataset which are readily available by setting the proper indexes.

We also provide a set of cover COCO images in /images.

<br><br>

### Imprint-Forgery Attack
Run this command to perform an Imprint-Forgery attack against SDXL
```bash
python run_imprint_forgery.py  --wm_type GS  --cover_image_path 'images/stalin.jpg'
```

Try the masking strategy like this
```bash
python run_imprint_forgery.py  --wm_type GS  --cover_image_path 'images/stalin.jpg'  --cover_mask_path 'masks/stalin.mask.png'
```

There should be directories with results popping up (./out/imprint-forgery).
Also, you should see output on the console.

For example, the command 
```bash
python run_imprint_forgery.py  --wm_type GS  --cover_image_path 'images/stalin.jpg'  --cover_mask_path 'masks/stalin.mask.png'
```
should produce outputs very similar to this:
```bash
Step 0, detection_success: False, bit accuracy: 0.54688, p_value: 0.0, psnr: 31.68439
  7%|█████████▉                                                                                                                                            | 10/151
Step 10, detection_success: False, bit accuracy: 0.68359, p_value: 0.0, psnr: 31.15294
 13%|███████████████████▊                                                                                                                                  | 20/151
Step 20, detection_success: True, bit accuracy: 0.76562, p_value: 0.0, psnr: 30.72373
 20%|█████████████████████████████▊                                                                                                                        | 30/151
Step 30, detection_success: True, bit accuracy: 0.83203, p_value: 0.0, psnr: 30.48064
...
```
Once the detection success is true, the Imprint-Forgery was successful and the target model recognized the image as watermarked.

Other params to try are:
```bash
python run_imprint_forgery.py --wm_type TR --modelid_target "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
python run_imprint_forgery.py --wm_type TR --modelid_target "black-forest-labs/FLUX.1-dev"  --guidance_scale_target 3.5 --num_inference_steps_target 20
python run_imprint_forgery.py --wm_type GS --modelid_target "black-forest-labs/FLUX.1-dev"  --guidance_scale_target 3.5 --num_inference_steps_target 20  --num_replications 256
```

See the python script for more parameter options.

There are multiple other cover images taken from MS COCO to try out in /images.

A full run with 150 steps might take up to 40 minutes, but first successful detection might occur much earlier depending on the target model.

<br><br>

### Imprint-Removal Attack

The Imprint-Removal attack works very similarly, with the results being saved into out/imprint-removal. However, you do not provide a cover image but either a target prompt index in which case a prompt from a list of the SD-Prompts dataset will be taken or you define a custom prompt.
```bash
python run_imprint_removal.py  --wm_type GS  --target_prompt_index 0
python run_imprint_removal.py  --wm_type GS  --target_prompt 'cat holding a sign that say I am watermarked'
python run_imprint_removal.py  --wm_type GS  --target_prompt 'cat holding a sign that say I am watermarked' --modelid_target "black-forest-labs/FLUX.1-dev"  --guidance_scale_target 3.5 --num_inference_steps_target 20  --num_replications 256
```

Outputs will look the same, however, the attack was successful once the detection success turns to False.

For example, the command 
```bash
python run_imprint_removal.py  --wm_type GS  --target_prompt_index 0
```
should produce outputs like this:
```bash
Step 0, detection_success: True, bit accuracy: 1.00000, p_value: 0.0, psnr: 34.76470
  7%|███████▍                                                                                                         | 10/151
Step 10, detection_success: True, bit accuracy: 0.89844, p_value: 0.0, psnr: 32.56701
 13%|██████████████▉                                                                                                  | 20/151
Step 20, detection_success: False, bit accuracy: 0.66406, p_value: 0.0, psnr: 31.43231
 20%|██████████████████████▍                                                                                          | 30/151
Step 30, detection_success: False, bit accuracy: 0.63281, p_value: 0.0, psnr: 30.93994
...
```
A full run with 150 steps might take up to 40 minutes, but first successful detection evasion might occur much earlier depending on the target model.

<br><br>

### Reprompting Attack

Again, you do not provide a cover image, but a target prompt index (in which case a prompt from a list of the SD-Prompts dataset will be taken, or you define a custom prompt) and an attacker prompt index (in which case a prompt from a list of the I2P-Prompts dataset will be taken, or you define a custom prompt).
```bash
python run_reprompting.py  --wm_type GS --target_prompt_index 0 --attacker_prompt 0
python run_reprompting.py  --wm_type GS --target_prompt "cat holding a sign that says 'I am watermarked'" --attacker_prompt "a man screaming locked inside a glass jar, photorealistic, hdr, studio photography "
```

The Reprompting attack is simpler.
There will be results saved into out/reprompt. Only one benign image and a harmful iamge.
The console will show the success of the watermark detection in the benign and the harmful, reprompted image.

The output for this call
```bash
python run_reprompting.py  --wm_type GS --target_prompt_index 0 --attacker_prompt_index 0
```
should look very similar to this:

```bash
phase 1: generate target image
Loading pipeline components...: 100%|████████████████████████████████████████████████████████████████████████████| 7/7
(Benign image) detection_success: True, bit accuracy: 1.00000, p_value: 0.0, psnr: inf
phase 2: invert using attacker model
Loading pipeline components...: 100%|████████████████████████████████████████████████████████████████████████████| 6/6
phase 3: generate attacker image
phase 4: invert using target model and verify watermark
(Harmful image) detection_success: True, bit accuracy: 0.96484, p_value: 0.0, psnr: -1.00000
```

A full run will only take about 30 seconds.
As some readers might not want to see very harmful content (blood etc.), we added only less harmfull prompts from the I2P dataset.
You can also just set custom prompts with --target_prompt and --attacker_prompt.
