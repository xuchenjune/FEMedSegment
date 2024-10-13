from models.FEMed.modeling_sam import SamModel
from transformers import SamProcessor
import torch
import monai
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import glob
import argparse
from utils.log import setup_logging
import datetime
from utils.utils import get_bounding_box
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from statistics import mean
import matplotlib.patches as patches
from tqdm import tqdm
import matplotlib.colors as colors
import time
import re
import random
from models.SAMDataset import SAMDataset
import torch.nn as nn
from utils.utils import get_latest_checkpoint, calculate_percase
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="FEMedSegment"
)

def test(agrs):
    random.seed = agrs.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = ['fold_1', 'fold_2', 'fold_3','fold_4', 'fold_5']
    data_types = ['2d_images', '2d_masks']
    parent_dir = os.path.dirname(os.path.dirname(args.main_dir))#os.path.dirname(os.path.dirname(args.main_dir))
    base_dir = os.path.join(parent_dir, 'KFoldContent', args.dataset)#, get_label_names(args['dataset'])[args['organ_index']])
    print(f'base_dir: {base_dir}')
    box_prompt_zoom = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=device, dtype=torch.float64)

    print("Model is training...")


    
    # offline mode
    local_path = "/home/download/sam-vit-base/"
    model = SamModel.from_pretrained(local_path) 
    processor = SamProcessor.from_pretrained(local_path)   
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'{torch.cuda.device_count()} GPUs are working parallel')
    data_paths = {}

    
    # Create directories and print the number of images and masks in each
    for dataset in datasets:
        for data_type in data_types:
            # Construct the directory path
            dir_path = os.path.join(base_dir, dataset, data_type)
            # Find images and labels in the directory
            files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
            # Store the image and label paths in the dictionary
            data_paths[f'{dataset}_{data_type}'] = files
    
    print('Number of patients', args.patients)
    print('Number of fold_1 images', len(data_paths['fold_1_2d_images']))
    print('Number of fold_1 masks', len(data_paths['fold_1_2d_masks']))
    print('Number of fold_2 images', len(data_paths['fold_2_2d_images']))
    print('Number of fold_2 masks', len(data_paths['fold_2_2d_masks']))
    print('Number of fold_3 images', len(data_paths['fold_3_2d_images']))
    print('Number of fold_3 masks', len(data_paths['fold_3_2d_masks']))
    print('Number of fold_4 images', len(data_paths['fold_4_2d_images']))
    print('Number of fold_4 masks', len(data_paths['fold_4_2d_masks']))
    print('Number of fold_5 images', len(data_paths['fold_5_2d_images']))
    print('Number of fold_5 masks', len(data_paths['fold_5_2d_masks']))

    print(processor)

    train_images = []
    train_masks = []

    test_images = None
    test_masks = None

    for key, value in data_paths.items():
        if args.test_fold in key:
            if 'images' in key:
                test_images = value
            elif 'masks' in key:
                test_masks = value
        else:
            if 'images' in key:
                train_images.extend(value)
            elif 'masks' in key:
                train_masks.extend(value)
                
    print(f'test_images:{len(test_images)}')   
    print(f'test_masks:{len(test_masks)}')
    print(f'test_images:{len(train_images)}')   
    print(f'test_masks:{len(train_masks)}')
    
    patient_ids = set()
    for path in train_images:
        filename = os.path.basename(path)
        if(args.dataset == 'BraTS2021'):
            patient_id = filename.split('_')[1]
            
        elif(args.dataset == 'LiTS17'):
            patient_id = re.search(r'volume-(\d{3})_\d+.nii', filename).group(1)
            
        patient_ids.add(patient_id)
        
    patient_ids = list(patient_ids)  
    
    n = args.patients  
    assert len(patient_ids) >= n + 1, "Not enough unique patientIDs available."

    selected_ids = random.sample(patient_ids, n + 2)
    print(f'selected_ids {selected_ids}')
    kshot_train_images = []
    kshot_train_masks = []
    val_images = []
    val_masks = []

    val_ids = [selected_ids.pop(), selected_ids.pop()]
    print(f'val_id {val_ids}')
    print(f'selected_ids {selected_ids}')
    if(args.dataset == 'BraTS2021'):
        for path in train_images:
            filename = os.path.basename(path)
            patient_id = filename.split('_')[1]
            
            if patient_id in selected_ids:
                kshot_train_images.append(path)
                mask_path = path.replace('flair', 'seg')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                kshot_train_masks.append(mask_path)
            elif patient_id in val_ids:
                val_images.append(path)
                mask_path = path.replace('flair', 'seg')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                val_masks.append(mask_path)
    elif(args.dataset == 'LiTS17'): #volume-005_312.nii.gz
        for path in train_images:
            filename = os.path.basename(path)
            patient_id = re.search(r'volume-(\d{3})_\d+.nii', filename).group(1)
            
            if patient_id in selected_ids:
                kshot_train_images.append(path)
                mask_path = path.replace('volume', 'segmentation')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                kshot_train_masks.append(mask_path)
            elif patient_id in val_ids:
                val_images.append(path)
                mask_path = path.replace('volume', 'segmentation')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                val_masks.append(mask_path) 
        
    print(f'kshot_train_images {len(kshot_train_images)}')
    print(f'kshot_train_masks {len(kshot_train_masks)}')
    print(f'val_images {len(val_images)}')
    print(f'val_masks {len(val_masks)}') 
    # create train and validation dataloaders
    train_dataset = SAMDataset(image_paths=kshot_train_images, mask_paths=kshot_train_masks, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(f'train_dataloader: {len(train_dataloader)}')
    val_dataset = SAMDataset(image_paths=val_images, mask_paths=val_masks, processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # # make sure we only compute gradients for mask decoder (encoder weights are frozen)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder"):# or name.startswith("prompt_encoder"):
            if "prompt_generator" in name:   
                print(f'param:{name},requires_grad = True ')
                param.requires_grad_(True)     
            elif "pyramid_adapter" in name:
                print(f'param:{name},requires_grad = True ')
                param.requires_grad_(True) 
            elif "feature_selectors" in name:
                print(f'param:{name},requires_grad = True ')
                param.requires_grad_(True)
            else:
                print(f'param:{name},requires_grad = False ')
                param.requires_grad_(False)  
                
        if name.startswith("prompt_encoder"):
            print(f'param:{name},requires_grad = False ')
            param.requires_grad_(False) 
            
        if name.startswith("mask_decoder"):
            print(f'param:{name},requires_grad = True ')
            param.requires_grad_(True) 

    # define training loop
    num_epochs = args.epochs

    model.to(device)

    # define optimizer
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

    #scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # define segmentation loss with sigmoid activation applied to predictions from the model
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # track mean train and validation losses
    mean_train_losses, mean_val_losses = [], []

    # create an artibarily large starting validation loss value
    best_val_loss = 100.0
    best_val_epoch = 0

    # set model to train mode for gradient updating
    model.train()

    start_time = time.time()
    # 90% From Image Size 1024 * 1024

    for epoch in range(num_epochs):
        # create temporary list to record training losses
        epoch_losses = []
        for i, batch in enumerate(tqdm(train_dataloader)):
            box_prompt = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=device, dtype=torch.float64)
            # forward pass
            outputs = model(pixel_values = batch["pixel_values"].to(device),
                        input_boxes = box_prompt,
                        multimask_output = False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
            
        # create temporary list to record validation losses
        val_losses = []
        
        
        # set model to eval mode for validation
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader):
                box_prompt_zoom = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=device, dtype=torch.float64)

                outputs = model(pixel_values = val_batch["pixel_values"].to(device),
                                input_boxes = batch["input_boxes"].cuda(),
                                multimask_output = False)                
            
                # calculate val loss
                predicted_val_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = val_batch["ground_truth_mask"].float().to(device)
                
                val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))

                val_losses.append(val_loss.item())
                
                y_pred = predicted_val_masks
                y = ground_truth_masks.unsqueeze(1)
            
        CurrentEpoch = epoch + 1
        print(f'EPOCH: {CurrentEpoch}, Train Mean loss: {mean(epoch_losses)}')

        mean_train_losses.append(mean(epoch_losses))
        mean_val_losses.append(mean(val_losses))
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f'Current Validation Mean Loss: {mean_val_loss}')
        
        # wandb logging
        wandb.log({"Train Mean loss": mean(epoch_losses), "Validation Mean Loss": mean_val_loss})

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_val_epoch = epoch

        if epoch - best_val_epoch > args.patience_epochs:
            print(f"Validation loss hasn't improved for {args.patience_epochs} epochs. Stopping early.")
            break

    print(mean_train_losses)

    total_time = time.time() - start_time
    print(f"Training completed in: {total_time:.2f} seconds")

    test_dataset = SAMDataset(image_paths=test_images, mask_paths=test_masks, processor=processor, test_mode= True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_dice, test_hd95, test_asd, test_iou = 0, 0, 0, 0
    step_count = 0
    # 90% From Image Size 1024 * 1024
    box_prompt_zoom = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=device, dtype=torch.float64)
    # # Iteratire through test images
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].cuda(),
                        input_boxes= box_prompt_zoom, #batch["input_boxes"].cuda(),
                        multimask_output=False)

            # compute loss
            ground_truth_masks = batch["ground_truth_mask"].float().cuda()
            
            # apply sigmoid
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            
            gt = ground_truth_masks[0].cpu().numpy()
            pred = medsam_seg
        
            try:
                dice, hd95, asd, iou = calculate_percase(gt, pred)
                test_dice += dice
                test_hd95 += hd95 
                test_asd += asd 
                test_iou += iou
                step_count = step_count + 1
                
                
                
                print(f'step: {step_count}, dice:{dice}, hd95:{hd95}, asd:{asd}, iou:{iou}')
            except ValueError as e:
                print(f"Error occurred:{e}" )
                print(f'Unrecognized GT: {batch["image_path"]}, mask:{batch["mask_path"]}')
                continue
            
                
            print(f'dice:{dice}, hd95:{hd95}, asd:{asd}, iou:{iou}, current mean dice {test_dice / step_count}')
            
        test_dice = test_dice / step_count
        test_hd95 =  test_hd95 / step_count
        test_asd = test_asd / step_count
        test_iou = test_iou / step_count

        print(f'Test Completed, {len(test_dataloader)} images have been predicted')
        print(f'dice:{test_dice}, hd95:{test_hd95}, asd:{test_asd},  iou:{test_iou}, step_count: {step_count}')
        wandb.log({
            "dice" : test_dice,
            "hd95" : test_hd95,
            "asd" : test_asd,
            "iou" : test_iou
        })        
    
    current_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"model_{args.dataset}_epoch{args.epochs}_{args.patients}Shots_{current_time}.pth"
    save_path = os.path.join(args.main_dir, 'checkpoints', filename)

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'loss': loss,
    }, save_path)

    print(f"Model saved to {filename}")    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FEMed")

    parser.add_argument("--dataset", type=str, choices=["BraTS2021", "LiTS17", "Kvasir-SEG"], help="Dataset Name")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")
    parser.add_argument("--patients", type=int, default=5, help="Patients")
    parser.add_argument("--patience_epochs", type=int, default=10, help="Patience_epochs")
    parser.add_argument("--checkpoint", type=str, default='', help="checkpoint file path")
    parser.add_argument("--test_fold", type=str, default='fold_1', help="test folder")


    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    main_dir = os.path.dirname(os.path.abspath(__file__))
    args.main_dir = main_dir
    print(args.main_dir)

    log_file = os.path.join(main_dir, f"run/test_log_{current_time}.txt")

    logger = setup_logging(log_file)

    logger.info("Experiment Settings:")
    logger.info(args)
    logger.info("Testing Start: %s", current_time)
    
    args.train_ratio = 0.7
    args.val_ratio = 0.1
    args.test_ratio = 0.2

    if args.dataset == "BraTS2021":
        args.organ_index = 1
        print(args.dataset)
    elif args.dataset == "LiTS17":
        print(args.dataset)
    elif args.dataset == "Kvasir-SEG":
        print(args.dataset)
    elif args.dataset == "DRIVE":
        print(args.dataset)
    else:
        raise ValueError("Unknown Dataset:", args.dataset)
    
    parent_dir = os.path.dirname(os.path.dirname(args.main_dir))
    base_dir = os.path.join(parent_dir, 'Content', args.dataset)
    datasets = ['train', 'val', 'test']
    data_types = ['2d_images', '2d_masks']
    
    test(args)