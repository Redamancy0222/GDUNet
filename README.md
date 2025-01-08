# Gradient domain model driven algorithm unfolding network for blind image deblurring - Official Pytorch Implementation

<p align="center">
<img src= "./img/network.jpeg" width="80%">

This repository provides the official PyTorch implementation of the following paper:

> Gradient domain model driven algorithm unfolding network for blind image deblurring
> 
> This paper has been submitted to ‘Neural Network’ and is currently under review...

---

## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Train](#Train)
3. [Test](#Test)



---

## Dependencies

- Python
- Pytorch (1.11)
- scikit-image
- opencv-python
- Tensorboard
- einops

---



---

## Train

bash train.sh

---

## Test

Realblur pre-trained model is available at https://drive.google.com/drive/folders/1l_R8_2UKfiQP_BYrgcQrmCBSe_ogwL41?usp=drive_link

bash test.sh

Output images will be saved in ``` results/model_name/dataset_name/``` folder.

We measured PSNR using [official RealBlur test code](https://github.com/rimchang/RealBlur#evaluation). You can get the PSNR we achieved by cloning and following the RealBlur repository.

---

## Acknowledgment: 
This code is based on the MGSTNet.

