import random
import PIL.ImageOps
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Config():
    training_dir = "../dataset-origin/mix_train/"
    testing_dir = "../dataset-origin/mix_test/"
    targetlist_dir = "C:/Users/yuxuan/1Project/SiameseNetwork_jupy/targetlist/targetlist"
    phish_dir = "C:/Users/yuxuan/1Project/SiameseNetwork_jupy/argetlist/phish"
    train_batch_size = 64
    train_number_epochs = 80
    learning_rate = 0.0001


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        open_image0 = Image.open(img0_tuple[0])
        open_image1 = Image.open(img1_tuple[0])
        rgb_img0 = open_image0.convert("RGB")
        rgb_img1 = open_image1.convert("RGB")
        img0 = rgb_img0
        img1 = rgb_img1
        if self.should_invert:
            img0 = PIL.ImageOps.invert(rgb_img0)
            img1 = PIL.ImageOps.invert(rgb_img1)
        if self.transform is not None:
            img0 = self.transform(rgb_img0)
            img1 = self.transform(rgb_img1)
        transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
        img0 = transform(img0)
        img1 = transform(img1)
        label = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        return img0_tuple[0], img1_tuple[0], img0, img1, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class TestYOLO_Dataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]
        img = Image.open(img_tuple[0])
        img_url = img_tuple[0]
        img_name = img_url[img_url.rfind("\\") + 1: img_url.rfind(".")]
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
