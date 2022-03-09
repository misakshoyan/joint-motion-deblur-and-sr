import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderTrain, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp_dir   = os.path.join(root_dir, "target_low")
        self.sharp4x_dir = os.path.join(root_dir, "target")

        self.data_len = len(os.listdir(self.blur_dir))
        print("Train dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # print("i = ", index)
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp   = os.path.join(self.sharp_dir, fname)
        fpath_sharp4x = os.path.join(self.sharp4x_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp_img = Image.open(fpath_sharp)
        sharp4x_img = Image.open(fpath_sharp4x)

        # return TF.to_tensor(blur_img.copy()), \
        #        TF.to_tensor(sharp_img.copy()), \
        #        TF.to_tensor(sharp4x_img.copy())

        C, H, W = (3, 180, 320)
        scale = 4
        GT_size = 256
        LQ_size = GT_size // scale

        # -1 for safety
        rnd_h = random.randint(0, max(0, H - LQ_size - 1))
        rnd_w = random.randint(0, max(0, W - LQ_size - 1))
        blur_img_patch = np.asarray(blur_img)[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        sharp_img_patch = np.asarray(sharp_img)[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        sharp4x_img_patch = np.asarray(sharp4x_img)[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]

        # print("before to_tensor: ", blur_img_patch.shape)

        blur_img_patch = TF.to_tensor(blur_img_patch)
        sharp_img_patch = TF.to_tensor(sharp_img_patch)
        sharp4x_img_patch = TF.to_tensor(sharp4x_img_patch)
        # # print("after to_tensor: ", blur_img_patch.shape)
        #
        #
        aug = random.randint(0, 8)
        # print(aug)
        # # Data Augmentations
        if aug==1:
            blur_img_patch = blur_img_patch.flip(1)
            sharp_img_patch = sharp_img_patch.flip(1)
            sharp4x_img_patch = sharp4x_img_patch.flip(1)
        elif aug==2:
            blur_img_patch = blur_img_patch.flip(2)
            sharp_img_patch = sharp_img_patch.flip(2)
            sharp4x_img_patch = sharp4x_img_patch.flip(2)
        elif aug==3:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2))
        elif aug==4:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=2)
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2), k=2)
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2), k=2)
        elif aug==5:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=3)
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2), k=3)
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2), k=3)
        elif aug==6:
            blur_img_patch = torch.rot90(blur_img_patch.flip(1),dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch.flip(1),dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch.flip(1),dims=(1,2))
        elif aug==7:
            blur_img_patch = torch.rot90(blur_img_patch.flip(2),dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch.flip(2),dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch.flip(2),dims=(1,2))

        return blur_img_patch, sharp_img_patch, sharp4x_img_patch


class DataLoaderTrainDeblur(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderTrainDeblur, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp_dir   = os.path.join(root_dir, "target_low")

        self.data_len = len(os.listdir(self.blur_dir))
        print("deblur Train dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp   = os.path.join(self.sharp_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp_img = Image.open(fpath_sharp)

        C, H, W = (3, 180, 320)
        LQ_size = 64

        # -1 for safety
        rnd_h = random.randint(0, max(0, H - LQ_size - 1))
        rnd_w = random.randint(0, max(0, W - LQ_size - 1))
        blur_img_patch = np.asarray(blur_img)[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        sharp_img_patch = np.asarray(sharp_img)[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

        # print("before to_tensor: ", blur_img_patch.shape)

        blur_img_patch = TF.to_tensor(blur_img_patch)
        sharp_img_patch = TF.to_tensor(sharp_img_patch)
        # # print("after to_tensor: ", blur_img_patch.shape)
        #
        #
        aug = random.randint(0, 8)
        # # Data Augmentations
        if aug==1:
            blur_img_patch = blur_img_patch.flip(1)
            sharp_img_patch = sharp_img_patch.flip(1)
        elif aug==2:
            blur_img_patch = blur_img_patch.flip(2)
            sharp_img_patch = sharp_img_patch.flip(2)
        elif aug==3:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2))
        elif aug==4:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=2)
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2), k=2)
        elif aug==5:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=3)
            sharp_img_patch = torch.rot90(sharp_img_patch,dims=(1,2), k=3)
        elif aug==6:
            blur_img_patch = torch.rot90(blur_img_patch.flip(1),dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch.flip(1),dims=(1,2))
        elif aug==7:
            blur_img_patch = torch.rot90(blur_img_patch.flip(2),dims=(1,2))
            sharp_img_patch = torch.rot90(sharp_img_patch.flip(2),dims=(1,2))

        return blur_img_patch, sharp_img_patch


class DataLoaderVal(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderVal, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp4x_dir = os.path.join(root_dir, "target")

        self.data_len = len(os.listdir(self.blur_dir))
        print("test dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp4x = os.path.join(self.sharp4x_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp4x_img = Image.open(fpath_sharp4x)

        return TF.to_tensor(blur_img.copy()), \
               TF.to_tensor(sharp4x_img.copy())


class DataLoaderValDeblur(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderValDeblur, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp_dir = os.path.join(root_dir, "target_low")

        self.data_len = len(os.listdir(self.blur_dir))
        print("deblur test dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp = os.path.join(self.sharp_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp_img = Image.open(fpath_sharp)

        return TF.to_tensor(blur_img.copy()), \
               TF.to_tensor(sharp_img.copy())


class DataLoaderValJoint(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderValJoint, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp_dir = os.path.join(root_dir, "target_low")
        self.sharp4x_dir = os.path.join(root_dir, "target")

        self.data_len = len(os.listdir(self.blur_dir))
        print("test dateloader size (including target_low images): ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp = os.path.join(self.sharp_dir, fname)
        fpath_sharp4x = os.path.join(self.sharp4x_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp_img = Image.open(fpath_sharp)
        sharp_img4x = Image.open(fpath_sharp4x)

        return TF.to_tensor(blur_img.copy()), \
               TF.to_tensor(sharp_img.copy()), \
               TF.to_tensor(sharp_img4x.copy())





class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename
