# IrwGAN (ICCV2021): [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Unaligned_Image-to-Image_Translation_by_Learning_to_Reweight_ICCV_2021_paper.pdf)

## [Update] 11/16/2021 Code is pushed, selfie-anime-danbooru dataset released.


## Dataset
**[Selfie2anime-Danbooru](https://drive.google.com/file/d/1jWjBygCJo5xrorIRJ8g5TprY69nnQuHY/view?usp=sharing)**

## Trained models and generated images
**[&#x1F34F;IrwGAN](https://junyanz.github.io/CycleGAN/) |  [Baseline](https://arxiv.org/pdf/1703.10593.pdf) |  [CycleGAN](https://github.com/junyanz/CycleGAN) |
[MUNIT](https://www.tensorflow.org/tutorials/generative/cyclegan) | [GcGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) | [NICE-GAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)**


### Usage

- Training:
```bash
python main.py --dataroot=datasets/selfie2anime-danbooru
```
- Resume:
```bash
python main.py --dataroot=datasets/selfie2anime-danbooru --phase=resume
```
- Test:
```bash
python main.py --dataroot=datasets/selfie2anime-danbooru --phase=test
```
