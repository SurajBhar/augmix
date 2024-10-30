# Title

"Performance evaluation of Resnet18 & ConvNeXt_tiny on OOD (Out of Distribution) Robustness using AugMix"

# AugMix

<img align="center" src="assets/augmix.gif" width="750">

## Introduction

AugMix is a data processing technique that mixes augmented images and
enforces consistent embeddings of the augmented images, which results in
increased robustness and improved uncertainty calibration. AugMix does not
require tuning to work correctly, as with random cropping or CutOut, and thus
enables plug-and-play data augmentation. AugMix significantly improves
robustness and uncertainty measures on challenging image classification
benchmarks, closing the gap between previous methods and the best possible
performance by more than half in some cases. Using AugMix, researchers obtained
state-of-the-art on ImageNet-C, ImageNet-P and in uncertainty estimation when
the train and test distribution do not match.

For more details please see our [ICLR 2020 paper](https://arxiv.org/pdf/1912.02781.pdf).

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

2.  Download CIFAR-10-C and CIFAR-100-C datasets with:

    ```
    mkdir -p ./data/cifar
    curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    tar -xvf CIFAR-100-C.tar -C data/cifar/
    tar -xvf CIFAR-10-C.tar -C data/cifar/
    ```

3.  Download ImageNet-C with:

    ```
    mkdir -p ./data/imagenet/imagenet-c
    curl -O https://zenodo.org/record/2235448/files/blur.tar
    curl -O https://zenodo.org/record/2235448/files/digital.tar
    curl -O https://zenodo.org/record/2235448/files/noise.tar
    curl -O https://zenodo.org/record/2235448/files/weather.tar
    tar -xvf blur.tar -C data/imagenet/imagenet-c
    tar -xvf digital.tar -C data/imagenet/imagenet-c
    tar -xvf noise.tar -C data/imagenet/imagenet-c
    tar -xvf weather.tar -C data/imagenet/imagenet-c
    ```

## Usage-for Official Augmix

The Jensen-Shannon Divergence loss term may be disabled for faster training at the cost of slightly lower performance by adding the flag `--no-jsd`.

Training recipes used in our paper:

WRN: `python cifar.py`

AllConv: `python cifar.py -m allconv`

ResNeXt: `python cifar.py -m resnext -e 200`

DenseNet: `python cifar.py -m densenet -e 200 -wd 0.0001`

ResNet-50: `python imagenet.py <path/to/imagenet> <path/to/imagenet-c>`

## Pretrained weights

Weights for a ResNet-50 ImageNet classifier trained with AugMix for 180 epochs are available
[here](https://drive.google.com/file/d/1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF/view?usp=sharing).

This model has a 65.3 mean Corruption Error (mCE) and a 77.53% top-1 accuracy on clean ImageNet data.

## Usage- for this project

Training recipes used in this study:

WRN: `python cifar.py`

Resnet18(pretrained): `python cifar.py -m resnet18 -pt -op AdamW -sc CosineAnnealingLR`

Resnet18(not-pretrained): `python cifar.py -m resnet18 -op AdamW -sc CosineAnnealingLR`

ConvNeXt_tiny(pretrained): `python cifar.py -m convnext_tiny -pt -op AdamW -sc CosineAnnealingLR`

ConvNeXt_tiny(not-pretrained): `python cifar.py -m convnext_tiny -op AdamW -sc CosineAnnealingLR`

For more details regarding usage you can refer to run-jobscript.sh file.

## Results

<img align="center" src="assets/Table4.png" width="750">
<img align="center" src="assets/Matplotlib_plot.png" width="750">

## Conclusion

The Resnet18 (not-pretrained) model trained on CIFAR-10 and optimized using SGD and Lambda learning rate scheduler performed best on the CIFAR-10 and
CIFAR-10-C datasets, with a clean error of 11.15 and mCE of 16.837, respectively. When trained using AdamW optimizer and CosineAnnealingLR Scheduler
with hyperparameter values of learning rate=0.001, weight decay=0.0001, and epochs=100, Resnet18(pretrained) model outperforms these results with a Test error of 11.06 and mCE of 16.705. This leads to the conclusion that when properly tuned, the AdamW optimizer in conjunction with the CosineAnnealingLR scheduler outperforms the Stochastic Gradient Descent optimizer in conjunction with the LambdaLR scheduler. However, the trade-off between corruption and perturbation errors is still there and is evident from the outputs of mean Flip Probability values. Irrespective of the methods of model optimization used in this study, the mean Flip probability value is the same for all model comparisons when compared as Resnet18 (SGD) v/s Resnet18(AdamW). It leads to the conclusion that optimization techniques do not affect perturbations, but this is not true for corruptions as illustrated in Figure 3.

Furthermore, the ConvNeXt-tiny model (with pretrained weights) outperforms other models only on Clean and mean Corruption errors. For the mean Flip probability, the results show that a model’s performance on mean Corruption error is poorer than its performance on mean Flip probability, indicating a trade-off between mean corruption error and mean-flip probability. On the ConvNeXt tiny model with pretrained weights, a separate experiment was run with the final settings of experiment 2, except for a larger batch size of 256. This experiment produced relatively good results in a shorter amount of time, with clean error = 5.94 and mean-Corruption error = 12.324. This shows a trade off between efficiency of the model and compute time.

## Citation

If you find Augmix useful for your work, please consider citing

```
@article{hendrycks2020augmix,
  title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

