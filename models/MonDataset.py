import torch
from PIL import Image
import matplotlib.pyplot as plt
import PIL.Image
from utils.log import setup_logging
import datetime
from utils.utils import get_bounding_box
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms



class RandomCropShorterEdge(object):
    def __call__(self, img):
        width, height = img.size
        short_edge = min(width, height)
        # Create a random crop transform
        random_crop = transforms.CenterCrop(short_edge)
        return random_crop(img)
    
transform = transforms.Compose([
    RandomCropShorterEdge(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class MonDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert("L")
        mask = PIL.Image.open(self.mask_paths[idx]).convert("L")

        image = self.transform(image)
        
        mask = self.transform(mask)
        
        image = image[0].numpy()
        
        image = (image * 255).astype(np.uint8)

        ground_truth_mask = mask.squeeze()
        
        # # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image))
        
        # # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)
 
        ground_truth_mask[ground_truth_mask < 0] = 1
        
        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=None, return_tensors="pt",do_rescale=False)

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = ground_truth_mask.to(dtype=torch.int8)#torch.from_numpy(ground_truth_mask.numpy().astype(np.int8))

        inputs['image_path'] = self.image_paths[idx]
        inputs['mask_path'] = self.mask_paths[idx]
        return inputs
