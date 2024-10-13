import numpy as np
from medpy import metric
from tqdm import tqdm
import pickle
import cv2
from skimage import measure
import glob
import os

def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''
    # if there is no mask in the array, set bbox to image size
    if len(np.unique(ground_truth_map)) > 1:
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))
        
        bbox = [x_min, y_min, x_max, y_max]
        
        return bbox
    else:
        return [0, 0, 256, 256] # if there is no mask in the array, set bbox to image size
    

def calculate_percase(pred, gt):
    pred[pred > 0.5] = 1
    gt[gt > 0.5] = 1
    
    if pred.sum()> 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        
        # Hausdorff Distance
        hd95 = metric.binary.hd95(pred, gt)
        
        asd = metric.binary.asd(pred, gt)
        
        iou = metric.binary.jc(pred, gt)
        
        return dice, hd95, asd, iou
    elif gt.sum() == 0:
        print('gt is null')
        raise ValueError("Ground truth has no target")
    # No output from Model
    elif pred.sum() == 0:    
        return 0, 0, 0, 0
    # Other exceptions : Ground truth = null
    else:  
        raise ValueError("Unknown Exceptions")
    
def calculate_percase_with_mask(pred, gt, mask=None):
    pred[pred > 0.5] = 1
    gt[gt > 0.5] = 1
    
    print(pred.shape, gt.shape, mask.shape)

    if mask:
        pred = pred * mask
    
    if pred.sum()> 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        
        # Hausdorff Distance
        hd95 = metric.binary.hd95(pred, gt)
        
        asd = metric.binary.asd(pred, gt)
        
        iou = metric.binary.jc(pred, gt)
        
        return dice, hd95, asd, iou
    elif gt.sum() == 0:
        print('gt is null')
        raise ValueError("Ground truth has no target")
    # No output from Model
    elif pred.sum() == 0:    
        return 0, 0, 0, 0
    # Other exceptions : Ground truth = null
    else:  
        raise ValueError("Unknown Exceptions")
    

def IoU(result, reference):

    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        iou = intersection / float(size_i1 + size_i2 - intersection)
    except ZeroDivisionError:
        iou = 0.0
    
    return iou

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    
    height, width, _ = image.shape
    
    if width > height:
        start = (width - height) // 2
        cropped_image = image[:, start:start+height]
    elif height > width:
        start = (height - width) // 2
        cropped_image = image[start:start+width, :]
    else:
        cropped_image = image
    
    processed_image = cv2.resize(cropped_image, (target_size, target_size))
    
    return processed_image


def get_latest_checkpoint(dataset_name, checkpoint_dir):
    checkpoint_pattern = f"{dataset_name}_weights_*.pth"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))

    if not checkpoint_files:
        return None

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    return latest_checkpoint