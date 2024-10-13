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


from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
)

class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, data_name, test_mode=False):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.test_mode = test_mode
        self.data_name = data_name
        self.transforms = transforms = Compose([
            
            # load .nii or .nii.gz files
            LoadImaged(keys=['img', 'label']),
            
            # add channel id to match PyTorch configurations
            EnsureChannelFirstd(keys=['img', 'label']),
            
            # reorient images for consistency and visualization
            Orientationd(keys=['img', 'label'], axcodes='RA'),
            
            # resample all training images to a fixed spacing
            #Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),
            
            # rescale image and label dimensions to 256x256 
            CenterSpatialCropd(keys=['img', 'label'], roi_size=(256,256)),
            
            # scale intensities to 0 and 255 to match the expected input intensity range
            ScaleIntensityRanged(keys=['img'], a_min=-1000, a_max=2000, 
                         b_min=0.0, b_max=255.0, clip=True), 
            
            ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, 
                         b_min=0.0, b_max=255.0, clip=True), 

            SpatialPadd(keys=["img", "label"], spatial_size=(256,256))
        ])

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
        if self.data_name == "BraTS2021" or self.data_name == "LiTS17":
            # convert the grayscale array to RGB (3 channels)
            array_rgb = np.dstack((image, image, image))
            # in this dataset, the contours are -1 so we change them to 1 for label and 0 for background
            ground_truth_mask[ground_truth_mask < 0] = 1            
        else:
            # RGB imgaes
            array_rgb = image.transpose(1,2,0)
            ground_truth_mask = ground_truth_mask[0]
            ground_truth_mask[ground_truth_mask > 0] = 1 
        
        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)
        
        # get bounding box prompt (returns xmin, ymin, xmax, ymax)
        
        prompt = get_bounding_box(ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))
        inputs['image_path'] = image_path
        inputs['mask_path'] = mask_path
        return inputs