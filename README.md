# joint-motion-deblur-and-sr
## Single Image Joint Motion Deblurring and Super-Resolution Using the Multi-Scale Channel Attention Modules
The paper is under review in the journal of [Mathematical Problems of Computer Science](http://mpcs.sci.am/index.php/mpcs)

## Installation
This repository is built in PyTorch 1.8.1 and tested on Ubuntu 18.04 environment.
The implementation is based on [MPRNet](https://github.com/swz30/MPRNet), [TLSC](https://github.com/megvii-research/tlsc).  
Please follow these instructions:
1. Clone the repository
```
git clone git@github.com:misakshoyan/joint-motion-deblur-and-sr.git
cd joint-motion-deblur-and-sr
```

2. Create conda environment
```
conda create -n pytorch_181 python=3.7
conda activate pytorch_181
```

3. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install yacs tensorboard natsort opencv-python tqdm scikit-image
pip uninstall pillow
conda install pillow=8.4
```

## Data Preparation

The [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset is used to train and evaluate the model.  
1. Create training and validation directories:  
   ```
   mkdir dataset; cd dataset
   mkdir train; cd train
   mkdir input
   mkdir target
   mkdir target_low; cd ../
   
   mkdir validation; cd validation
   mkdir input
   mkdir target
   ```
2. Please download and unzip the following sets into *dataset* folder:  
   - Training: *train_blur_bicubic, train_sharp_bicubic and train_sharp*
   - Validation: *val_blur_bicubic and val_sharp*
   - Validation (Val300, prepared): *[here](https://drive.google.com/drive/folders/1JX3_P1r08ME9_MusSflmWRNVaqv0gp0c?usp=sharing)*
3. Copy and rename the images as {0-N}.png by the following bash command:  
   ```
   cd dataset/train_blur_bicubic/train/train_blur_bicubic/X4
   j=0;for i in */*.png; do cp "$i" ../../../../train/input/"$j".png; let j=j+1;done
   
   cd dataset/train_sharp_bicubic/train/train_sharp_bicubic/X4/
   j=0;for i in */*.png; do cp "$i" ../../../../train/target_low/"$j".png; let j=j+1;done
   
   cd dataset/train_sharp/train/train_sharp/
   j=0;for i in */*.png; do cp "$i" ../../../train/target/"$j".png; let j=j+1;done
   
   cd dataset/val_blur_bicubic/val/val_blur_bicubic/X4
   j=0;for i in */*.png; do cp "$i" ../../../../validation/input/"$j".png; let j=j+1;done
   
   cd dataset/val_sharp/val/val_sharp
   j=0;for i in */*.png; do cp "$i" ../../../validation/target/"$j".png; let j=j+1;done
   ```
   Val300 is already processed.
4. Crop REDS validation images to be multiple of 8 since MPRNet relies on multi-patch encoder-decoder architecture (320x180 -> 320x176, 1280x720 -> 1280x704):
   ```
   cd dataset/validation/input
   mkdir _cropped
   # WIDTHxHEIGHT+left+top
   find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -crop 320x176+0+2 "{}" _cropped/"{}"
   rm *.png
   cd _cropped
   mv *.png ../
   rm -rf _cropped
   
   cd dataset/validation/target
   mkdir _cropped
   find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -crop 1280x704+0+8 "{}" _cropped/"{}"
   rm *.png
   cd _cropped
   mv *.png ../
   rm -rf _cropped
   ```
   Val300 is already processed.

## Training
A three-phase training strategy is employed:
1. Train the deblurring module:
   ```
   python train_deblur.py
   ```
   Copy the best model as _latest.pth. 
2. Train the super-resolution module using the pre-traind frozen deblurring module as feature extractor:
   ```
   python train_deblurSR_freezedDB.py
   ```
   Copy the best model as _latest.pth.
3. Train the whole network jointly with unfrozen deblurring module:
   ```
   python train_deblurSR.py
   ```
Change the training parameters from training.yml for each phase.

## Evaluation
### Download the pre-trained [model](https://drive.google.com/drive/folders/1JX3_P1r08ME9_MusSflmWRNVaqv0gp0c?usp=sharing) and place it in *pre-trained-model* directory
Run the following command:
```
python test.py --input_dir dataset/Val300 --weights pre-trained-model/model_best.pth --result_dir dir-to-save-result
```
Use *--flip_test* option to test with self-ensemble strategy.