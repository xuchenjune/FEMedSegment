# from models.FEMed.modeling_sam import SamModel
from transformers import SamProcessor
import torch
import monai
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import glob
import argparse
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
import torch.nn.functional as F
from torchvision.utils import save_image

# inference func for MedSAM and SAM-Med2D
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    # low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_logits = F.interpolate(
        low_res_logits,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  

    outputs = {}
    outputs['pred_masks'] = low_res_logits
    outputs['pred_logits'] = low_res_logits
    return outputs

def build_model(args):
    if args.model == "FEMed":
        from models.FEMed.modeling_sam import SamModel
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        return model
    
    elif args.model == "SAM":
        from transformers import SamModel 
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        return model
    
    elif args.model == "MedSAM":
        from segment_anything import sam_model_registry
        checkpoint = "/home/Codes/FEMedSegment/baselines/MedSAM/medsam_vit_b.pth"
        model = sam_model_registry["vit_b"](checkpoint)
        return model
    
    elif args.model == "SAM-Med2D":
        from baselines.SAM_Med2D.segment_anything import sam_model_registry
        args.sam_checkpoint = "/home/Codes/FEMedSegment/baselines/SAM_Med2D/sam-med2d_b.pth"
        args.image_size = 256
        args.encoder_adapter = True
        args.boxes_prompt = True
        model = sam_model_registry["vit_b"](args)    
        return model
    
def build_dataset(args):
    datasets = ['fold_1', 'fold_2', 'fold_3','fold_4', 'fold_5']
    data_types = ['2d_images', '2d_masks']
    parent_dir = os.path.dirname(os.path.dirname(args.main_dir))
    base_dir = os.path.join(parent_dir, 'KFoldContent', args.dataset)
    print(f'base_dir: {base_dir}')
    
    train_with_gt_prompts = args.train_with_gt_prompts
    inference_with_gt_prompts = args.inference_with_gt_prompts  
    wandb_logger = args.wandb  
    
    print(f"Training with gt prompts {train_with_gt_prompts}")
    print(f"Inference with gt prompts {inference_with_gt_prompts}")

    print(f"Training with {args.model}...")
    
    if wandb_logger:
        wandb.init(
            # set the wandb project where this run will be logged
            project="FEMedSegment"
        )        

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")  
    
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
            if args.dataset == 'BraTS2021':
                files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
            elif args.dataset == 'Kvasir-SEG':
                files = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
            elif args.dataset == 'LiTS17':
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
    print(f'train_images:{len(train_images)}')   
    print(f'train_masks:{len(train_masks)}')
    
    patient_ids = set()
    for path in train_images:
        filename = os.path.basename(path)
        if(args.dataset == 'BraTS2021'):
            patient_id = filename.split('_')[1]
            
        elif(args.dataset == 'LiTS17'):
            patient_id = filename.split("-")[-1].split("_")[0]
            
        elif(args.dataset == 'Kvasir-SEG'):
            patient_id = filename.split('.')[0]
            
        patient_ids.add(patient_id)
        
    patient_ids = list(patient_ids)  
    # ensure the files order remains the same
    patient_ids.sort()  
    
    n = args.patients  
    assert len(patient_ids) >= n + 1, "Not enough unique patientIDs available."

    random.seed(args.seed)
    selected_ids = random.sample(patient_ids, n + 2)
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
    elif(args.dataset == 'LiTS17'): #Example: volume-005_312.nii.gz
        for path in train_images:
            filename = os.path.basename(path)
            patient_id = filename.split("-")[-1].split("_")[0]
            
            if patient_id in selected_ids:
                kshot_train_images.append(path)
                # Build mask path
                mask_path = path.replace('volume', 'segmentation')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                kshot_train_masks.append(mask_path)
            elif patient_id in val_ids:
                val_images.append(path)
                mask_path = path.replace('volume', 'segmentation')
                mask_path = mask_path.replace('2d_images', '2d_masks')
                val_masks.append(mask_path) 

    elif(args.dataset == 'Kvasir-SEG'): #Example: cju3ykamdj9u208503pygyuc8.jpg
        for path in train_images:
            filename = os.path.basename(path)
            patient_id = patient_id = filename.split('.')[0]
            
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
    train_dataset = SAMDataset(image_paths=kshot_train_images, mask_paths=kshot_train_masks, processor=processor, data_name=args.dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(f'train_dataloader: {len(train_dataloader)}')
    val_dataset = SAMDataset(image_paths=val_images, mask_paths=val_masks, processor=processor, data_name=args.dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) 
    
    test_dataset = SAMDataset(image_paths=test_images, mask_paths=test_masks, processor=processor, test_mode=True, data_name=args.dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
    
    return train_dataloader, val_dataloader, test_dataloader 

def reveal_mask_decoder(model):
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
    return model    

def train_step(args, model, train_dataloader, loss, optimizer):
    
    model.train()
    epoch_losses = []
    for i, batch in enumerate(train_dataloader):
        box_prompt = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=model.device, dtype=torch.float64)   
        if args.model in ["MedSAM", "SAM-Med2D"]:
            if args.model == "MedSAM":
                H, W = 1024, 1024
            else:
                H, W = 256, 256
            batch["pixel_values"] = F.interpolate(batch["pixel_values"], (H, W))
            outputs = medsam_inference(model, model.image_encoder(batch["pixel_values"].to(model.device)), box_prompt, H, W)
        else:
            outputs = model(pixel_values = batch["pixel_values"].to(model.device), 
                            input_boxes = box_prompt, 
                            multimask_output = False)                   
                
        # compute loss
        if args.model in ["MedSAM", "SAM-Med2D"]:
            # all GTs are tested in 256x256 scale
            predicted_masks = F.interpolate(outputs['pred_logits'], (256, 256))
            
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(model.device)
        loss_epoch = loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss_epoch.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss_epoch.item())
    
    return epoch_losses

@torch.no_grad()
def validation_step(args, model, val_dataloader, loss):
    model.eval()
    epoch_losses = []
    for i, batch in enumerate(val_dataloader):
        # TODO baseline method implementation
        if args.train_with_gt_prompts:
            outputs = model(pixel_values = batch["pixel_values"].to(model.device), 
                            input_boxes = batch["input_boxes"].to(model.device), 
                            multimask_output = False) 
        else:               
            # TODO this is a fixed ratio bbox prompt, adjust later!!
            box_prompt = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=model.device, dtype=torch.float64)   
            if args.model in ["MedSAM", "SAM-Med2D"]:
                if args.model == "MedSAM":
                    H, W = 1024, 1024
                else:
                    H, W = 256, 256
                batch["pixel_values"] = F.interpolate(batch["pixel_values"], (H, W))
                outputs = medsam_inference(model, model.image_encoder(batch["pixel_values"].to(model.device)), box_prompt, H, W)
            else:
                outputs = model(pixel_values = batch["pixel_values"].to(model.device), 
                                input_boxes = box_prompt, 
                                multimask_output = False)                   
                
        # compute loss
        if args.model in ["MedSAM", "SAM-Med2D"]:
            # all GTs are tested in 256x256 scale
            predicted_masks = F.interpolate(outputs['pred_logits'], (256, 256))
            
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(model.device)
        
        loss_epoch = loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        
        epoch_losses.append(loss_epoch.item())
    
    return epoch_losses 

@torch.no_grad()
def test_step(args, model, test_dataloader):
    
    model.eval()
    test_dice, test_hd95, test_asd, test_iou = 0, 0, 0, 0
    step_count = 0       

    progress_bar = tqdm(total=len(test_dataloader), desc="Testing")     
    
    for i, batch in enumerate(test_dataloader):
        box_prompt = torch.tensor([[[51.2, 51.2, 972.8, 972.8]]], device=model.device, dtype=torch.float64)   
        
        if args.model in ["MedSAM", "SAM-Med2D"]:
            if args.model == "MedSAM":
                H, W = 1024, 1024
            else:
                H, W = 256, 256
            batch["pixel_values"] = F.interpolate(batch["pixel_values"], (H, W))
            outputs = medsam_inference(model, model.image_encoder(batch["pixel_values"].to(model.device)), box_prompt, H, W)
        else:
            outputs = model(pixel_values = batch["pixel_values"].to(model.device), 
                            input_boxes = box_prompt, 
                            multimask_output = False)  

        ground_truth_masks = batch["ground_truth_mask"].float().cuda()
        
        # apply sigmoid
        if args.model in ["MedSAM", "SAM-Med2D"]:
            medsam_seg_prob = torch.sigmoid(F.interpolate(outputs['pred_logits'], (256, 256)))
        else:
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        
        gt = ground_truth_masks[0].cpu().numpy()
        pred = medsam_seg

        try:
            dice, hd95, asd, iou = calculate_percase(pred, gt)
            test_dice += dice
            test_hd95 += hd95 
            test_asd += asd 
            test_iou += iou
            step_count = step_count + 1       
            
        except ValueError as e:
            continue
        
        
        progress_bar.update(1)
        progress_bar.set_postfix({"dice": dice, "hd95": hd95, "iou": iou})
        progress_bar.refresh()       
        
    test_dice = test_dice / step_count
    test_hd95 =  test_hd95 / step_count
    test_asd = test_asd / step_count
    test_iou = test_iou / step_count
    print(f'\n Test Completed, {len(test_dataloader)} images have been predicted')
    print(f'dice:{test_dice}, hd95:{test_hd95}, asd:{test_asd},  iou:{test_iou}, step_count: {step_count}') #jc:{test_jc} jc = iou        

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):
    set_seed(args.seed)
    model = build_model(args)

    train_dataloader, val_dataloader, test_dataloader = build_dataset(args)
    
    ####### hyperparameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = args.epochs
    model.to(device)
    ####### hyperparameter
    
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    train_loss = []
    val_loss = []
    best_val_loss = 100.0
    best_val_epoch = 0   

    progress_bar = tqdm(total=num_epochs, desc="Training")

    for epoch in range(num_epochs):
        epoch_train_loss = train_step(args, model, train_dataloader, seg_loss, optimizer)
        epoch_val_loss = validation_step(args, model, val_dataloader, seg_loss)

        train_loss.append(mean(epoch_train_loss))
        val_loss.append(mean(epoch_val_loss)) 

        progress_bar.update(1)
        progress_bar.set_postfix({"Train loss": mean(epoch_train_loss), "Val loss" : mean(epoch_val_loss)})
        progress_bar.refresh()        

        
        # early stop
        if mean(val_loss) < best_val_loss:
            best_val_loss = mean(val_loss)
            best_val_epoch = epoch
        if epoch - best_val_epoch > args.patience_epochs:
            print(f"Validation loss hasn't improved for {args.patience_epochs} epochs. Stopping early.")
            break 

    # test
    test_step(args, model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEMedSegmentation")
    parser.add_argument("--model", type=str, choices=["FEMed", "SAM", "MedSAM", "SAM-Med2D"], help="type of model")
    parser.add_argument("--dataset", type=str, choices=["BraTS2021", "LiTS17", "Kvasir-SEG"], help="Dataset Name")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")
    parser.add_argument("--patients", type=int, default=5, help="Patients")
    parser.add_argument("--patience_epochs", type=int, default=10, help="Patience_epochs")
    parser.add_argument("--test_fold", type=str, default='fold_1', help="test folder")
    parser.add_argument("--train_with_gt_prompts", action='store_true', help="trained with gt prompts")
    parser.add_argument("--inference_with_gt_prompts", action='store_true', help="trained with gt prompts")
    parser.add_argument("--wandb", action='store_true', help="logging with wandb")    
    
    args = parser.parse_args()
    
    main_dir = os.path.dirname(os.path.abspath(__file__))
    args.main_dir = main_dir
    
    main(args)
