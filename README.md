# AugMix

<img align="center" src="assets/augmix.gif" width="750">

## Objectives

The main tasks assigned for this study are:
1. The addition of neural networks "Resnet18 and ConvNeXt-tiny" to the
augmix code (not-pretrained and pre-trained).
2. Training of both neural networks on CIFAR-10 and Evaluation on CIFAR-
10, CIFAR-10-C, and CIFAR-10-P datasets with:
(a) AdamW optimizer and cosine annealing learning rate scheduler.
(b) SGD optimizer and lambda learning rate scheduler.
3. Hyperparameter tuning of convnext-tiny model to improve it’s performance

For more details please see our [ICLR 2020 paper](https://arxiv.org/pdf/1912.02781.pdf).

## Pseudocode

<img align="center" src="assets/pseudocode.png" width="750">

## Requirements

*   numpy>=1.15.0
*   Pillow>=6.1.0
*   torch>=1.2.0
*   torchvision==0.14.0

## Setup

1.  Install PyTorch and other required python libraries with:

    ```
    pip install -r requirements.txt
    ```

2.  Download CIFAR-10-C dataset with:

    ```
    mkdir -p ./data/cifar
    curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    tar -xvf CIFAR-10-C.tar -C data/cifar/
    ```

## Usage

The Jensen-Shannon Divergence loss term may be disabled for faster training at the cost of slightly lower performance by adding the flag `--no-jsd`.

Training recipes used in our paper:

WRN: `python cifar.py`

Resnet18: `python cifar.py -m resnet18 -pt -op AdamW -sc CosineAnnealingLR`

ConvNeXt_tiny: `python cifar.py -m convnext_tiny -op AdamW -sc CosineAnnealingLR`
