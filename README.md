# AugMix

<img align="center" src="assets/augmix.gif" width="750">


## Title
"Performance evaluation of Resnet18 & ConvNeXt_tiny on OOD (Out of Distribution) Robustness using AugMix"

## Objectives

The main tasks assigned for this study are:
1. The addition of neural networks "Resnet18 and ConvNeXt-tiny" to the
augmix code (not-pretrained and pre-trained).
2. Training of both neural networks on CIFAR-10 and Evaluation on CIFAR-
10, CIFAR-10-C, and CIFAR-10-P datasets with:
(a) AdamW optimizer and cosine annealing learning rate scheduler.
(b) SGD optimizer and lambda learning rate scheduler.
3. Hyperparameter tuning of convnext-tiny model to improve it’s performance

For more details please see [ICLR 2020 paper](https://arxiv.org/pdf/1912.02781.pdf) and the attached report in this repository "Bhardwaj-Suraj-1531066.pdf".

## Results

<img align="center" src="assets/Table4.png" width="750">
<img align="center" src="assets/Matplotlib_plot" width="750">

## Requirements
*   numpy>=1.15.0
*   Pillow>=6.1.0
*   torch>=1.2.0
*   torchvision==0.14.0

## Usage

Training recipes used in this study:

WRN: `python cifar.py`
Resnet18: `python cifar.py -m resnet18 -pt -op AdamW -sc CosineAnnealingLR`
Resnet18: `python cifar.py -m resnet18 -op AdamW -sc CosineAnnealingLR`
ConvNeXt_tiny: `python cifar.py -m convnext_tiny -pt -op AdamW -sc CosineAnnealingLR`
ConvNeXt_tiny: `python cifar.py -m convnext_tiny -op AdamW -sc CosineAnnealingLR`

For more details regarding usage you can refer to run-jobscript.sh file.
