# data_preprocessing.py
import os
import glob
import random
import torch
import numpy as np 
import SimpleITK as sitk
import argparse
from utils.log import setup_logging
import datetime
import shutil

def preprocess_data(args):
    random.seed = 42
    print("Preprocessing...")
    print("Parameters:", args.dataset)
    print(args.main_dir)
    
    parent_dir = os.path.dirname(os.path.dirname(args.main_dir))
    data_dir = os.path.join(parent_dir, 'Datasets', args.dataset)
    print(data_dir)
    if args.dataset == 'BraTS2021' :        
        images = sorted(glob.glob(os.path.join(data_dir, '**', '*flair.nii.gz'), recursive=True))
        labels = sorted(glob.glob(os.path.join(data_dir, '**', '*seg.nii.gz'), recursive=True))
    if args.dataset == 'LiTS17' : 
        images = sorted(glob.glob(os.path.join(data_dir, '*', 'volume-*.nii'), recursive=True))
        labels = sorted(glob.glob(os.path.join(data_dir, '*', 'segmentation-*.nii'), recursive=True))
    if args.dataset == 'Kvasir-SEG' :
        images = sorted(glob.glob(os.path.join(data_dir, 'images', '*.jpg'), recursive=True))
        labels = sorted(glob.glob(os.path.join(data_dir, 'masks', '*.jpg'), recursive=True))
    
    assert len(images) == len(labels)
    
    indices = np.arange(len(images))
    random.shuffle(indices)

    images = np.array(images)[indices].tolist()
    labels = np.array(labels)[indices].tolist()
    
    base_dir = os.path.join(parent_dir, 'KFoldContent', args.dataset)#, get_label_names(args['dataset'])[args['organ_index']]) 
    
    splits = split_indices(len(images), n_splits=5)

    for fold, train_test_index in enumerate(splits, start=1):
        print(f"Fold {fold}:")
        print("indices:", train_test_index)

        fold_dir = os.path.join(base_dir, f'fold_{fold}')
        images_dir = os.path.join(fold_dir, '2d_images')
        masks_dir = os.path.join(fold_dir, '2d_masks')
        print(images_dir, masks_dir)
    
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        
        for idx in train_test_index:   

            if args.dataset == 'Kvasir-SEG':
                shutil.copy2(images[idx], images_dir)
                shutil.copy2(labels[idx], masks_dir)
            else:

                image_dst = os.path.join(images_dir, os.path.basename(images[idx]))
                label_dst = os.path.join(masks_dir, os.path.basename(labels[idx]))     
                
                print(f'image_dst{image_dst}, label_dst{label_dst}')

                
                img = sitk.ReadImage(images[idx])
                mask = sitk.ReadImage(labels[idx])
                print('processing patient', idx, img.GetSize(), mask.GetSize())
                # Get the mask data as numpy array
                mask_data = sitk.GetArrayFromImage(mask)
                
                if args.dataset == 'BraTS2021':
                    # label0：BK, background
                    # label1：NT, necrotic tumor core
                    # label2：ED, peritumoral edema
                    # label4：ET, enhancing tumor
                    for i in range(0, img.GetSize()[2], args.slice_step):
                        #! If the mask slice is not empty, save the image and merge 1,2,4 to 5
                        if np.any(np.isin(mask_data[i, :, :], [1, 2, 4])): 
                            # Prepare the new ITK images
                            img_slice = img[:, :, i]  
                            # Merge labels 1, 2, 4 into label 5
                            new_array = np.where(np.isin(mask_data[i, :, :], [1, 2, 4]), 1, mask_data[i, :, :])
                            mask_slice = sitk.GetImageFromArray(new_array)
                            
                            # Define the output paths
                            img_slice_path = os.path.join(images_dir, f"{os.path.basename(image_dst).replace('.nii.gz', '')}_{i}.nii.gz")
                            mask_slice_path = os.path.join(masks_dir, f"{os.path.basename(label_dst).replace('.nii.gz', '')}_{i}.nii.gz")
                            # # Save the slices as NIfTI files
                            sitk.WriteImage(img_slice, img_slice_path)
                            sitk.WriteImage(mask_slice, mask_slice_path)       
                            
                elif args.dataset == 'LiTS17':
                # label 1：Liver
                # label 2：Tumor
                    for i in range(0, img.GetSize()[2], args.slice_step):
                        if np.any(mask_data[i, :, :] == 1):
                        # Prepare the new ITK images
                            img_slice = img[:, :, i]

                            new_array = np.where(mask_data[i, :, :] != 1, 0.0, mask_data[i, :, :])
                            mask_slice = sitk.GetImageFromArray(new_array)

                            # Define the output paths
                            img_slice_path = os.path.join(images_dir, f"{os.path.basename(image_dst).replace('.nii', '')}_{i}.nii.gz")
                            mask_slice_path = os.path.join(masks_dir, f"{os.path.basename(label_dst).replace('.nii', '')}_{i}.nii.gz")
                            
                            # # Save the slices as NIfTI files
                            sitk.WriteImage(img_slice, img_slice_path)
                            sitk.WriteImage(mask_slice, mask_slice_path)
    print("Preprocess Finished")
    return base_dir

def split_indices(data_size, n_splits=5):
    """Generate the index of n splits for mutually exclusive subsets"""
    indices = np.arange(data_size)
    split_size = data_size // n_splits
    remainder = data_size % n_splits
    splits = []
    
    start = 0
    for i in range(n_splits):
        end = start + split_size + (1 if i < remainder else 0)
        splits.append(indices[start:end])
        start = end
    return splits
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset", type=str, choices=["BraTS2021", "LiTS17", "Kvasir-SEG"], help="Dataset Name")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--slice_step", type=int, default=1, help="Window Slice Step Length")
    parser.add_argument("--patients", type=int, default=1, help="Example Count")

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    main_dir = os.path.dirname(os.path.abspath(__file__))
    args.main_dir = main_dir

    log_file = os.path.join(main_dir, f"run/dataset_log_{current_time}.txt")

    logger = setup_logging(log_file)

    logger.info("Experiment Setup:")
    logger.info(args)

    logger.info("Start: %s", current_time)
    
    args.train_ratio = 0.8
    args.test_ratio = 0.2
    
    if args.dataset == "BraTS2021":
        print(args.dataset)
    elif args.dataset == "LiTS17":
        print(args.dataset)
    elif args.dataset == "Kvasir-SEG":
        print(args.dataset)
    elif args.dataset == "DRIVE":
        print(args.dataset)
    else:
        raise ValueError("Unknown Dataset:", args.dataset)
    
    preprocess_data(args)