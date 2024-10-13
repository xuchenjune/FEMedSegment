import torch
import monai
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import argparse
from utils.log import setup_logging
import datetime
from utils.utils import get_bounding_box
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
#from monai.transforms import Compose, CenterSpatialCropd

import PIL.Image
class DRIVEDataset2(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'img': image_path, 'label': mask_path})

        # squeeze extra dimensions
        image = data_dict['img'].squeeze()
        ground_truth_mask = data_dict['label'].squeeze()

        # convert to int type for huggingface's models expected inputs
        image = image.astype(np.uint8)

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image))
        
        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)
        
        # get bounding box prompt (returns xmin, ymin, xmax, ymax)
        # in this dataset, the contours are -1 so we change them to 1 for label and 0 for background
        ground_truth_mask[ground_truth_mask < 0] = 1
        
        prompt = get_bounding_box(ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))

        return inputs

class RandomCropShorterEdge(object):
    def __call__(self, img):
        width, height = img.size
        short_edge = min(width, height)
        # Create a random crop transform
        random_crop = transforms.CenterCrop(short_edge)
        return random_crop(img)
    
       
# 定义CenterSpatialCropd变换
transform = transforms.Compose([
    RandomCropShorterEdge(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class DRIVEDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx])
        mask = PIL.Image.open(self.mask_paths[idx])

        image = self.transform(image)

        mask = self.transform(mask)
        
        image.to(torch.uint8)

        ground_truth_mask = mask.squeeze()
    
        ground_truth_mask[ground_truth_mask < 0] = 1
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=None, return_tensors="pt",do_rescale=False)

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = ground_truth_mask.to(dtype=torch.int8)#torch.from_numpy(ground_truth_mask.numpy().astype(np.int8))
        inputs['image_path'] = self.image_paths[idx]
        inputs['mask_path'] = self.mask_paths[idx]
        return inputs
    
class DRIVETestDataset(Dataset):
    def __init__(self, image_paths, mask_paths, cover_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.cover_paths = cover_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx])
        mask = PIL.Image.open(self.mask_paths[idx])
        cover = PIL.Image.open(self.cover_paths[idx])

        image = self.transform(image)

        mask = self.transform(mask)
        
        cover = self.transform(cover)
        
        image.to(torch.uint8)

        ground_truth_mask = mask.squeeze()
        
        ground_truth_mask[ground_truth_mask < 0] = 1

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=None, return_tensors="pt", do_rescale=False)

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = ground_truth_mask.to(dtype=torch.int8)#torch.from_numpy(ground_truth_mask.numpy().astype(np.int8))
        inputs['image_path'] = self.image_paths[idx]
        inputs['mask_path'] = self.mask_paths[idx]
        inputs['cover'] = cover
        return inputs