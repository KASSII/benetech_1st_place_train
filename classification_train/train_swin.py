import argparse
import json
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

import datasets
from transform import get_train_transforms, get_valid_transforms
from torch.cuda.amp import GradScaler, autocast
import json
import torch.optim as optim
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

class Config:
    class_num = 5
    size = 384
    batch_size = 16
    epochs = 15
    lr = 3e-5
    num_warmup_steps = 1000

    validation_step = 1000

    model = "swin_l_384"
    use_pretrained = True

    patience_num = 10

train_csv_list = [
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/extracted_clean_4fold.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [1,2,3],
            "val_folds": [0],
        },
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/generated_clean_4fold.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [0,1,2,3],
            "val_folds": [],
        },
        {
            "csv_path": "../datas/input/ICDAR_2022/useful/cleaned_fold_4.csv",
            "data_path": "../datas/input/ICDAR_2022/useful",
            "train_folds": [1,2,3],
            "val_folds": [0]
        },

        {
            "csv_path": "../datas/input/generated_synthetic/dot/data_categorical_v2/data_list_category_v2.csv",
            "data_path": "../datas/input/generated_synthetic/dot/data_categorical_v2",
            "train_folds": [1,2,3,4,5,6,7,8,9],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/generated_synthetic/dot/data_numerical/numerical_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/dot/data_numerical/",
            "train_folds": [1,2,3,4,5,6,7,8,9],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/generated_synthetic/horizontal_bar/data_Bartley/data_list.csv",
            "data_path": "../datas/input/generated_synthetic/horizontal_bar/data_Bartley",
            "train_folds": [1,2,3,4,5,6,7,8,9],
            "val_folds": [0]
        },
    ]

label_map = {
    "vertical_bar": 0,
    "horizontal_bar": 1,
    "line": 2,
    "dot": 3,
    "scatter": 4
}

def main():
    parser = argparse.ArgumentParser(description='Classification Train')
    parser.add_argument("--output", type=str, default=".", required=False)
    args = parser.parse_args()

    cfg = Config()

    ### DataLoder
    train_df_list = []
    val_df_list = []
    for train_csv_info in train_csv_list:
        df = pd.read_csv(train_csv_info["csv_path"])
        df["image_path"] = os.path.join(train_csv_info["data_path"], "images")

        train_folds = train_csv_info["train_folds"]
        train_df = df.query("fold in @train_folds")
        val_folds = train_csv_info["val_folds"]
        valid_df = df.query("fold in @val_folds")
        if len(train_df) > 0:
            train_df_list.append(train_df)
            print(f'Add Train -> {train_csv_info["csv_path"]}/fold:{train_folds}')
        if len(valid_df) > 0:
            val_df_list.append(valid_df)
            print(f'Add Valid -> {train_csv_info["csv_path"]}/fold:{val_folds}')

    train_df = pd.concat(train_df_list).reset_index(drop=True)
    valid_df = pd.concat(val_df_list).reset_index(drop=True)

    train_df["label"] = train_df["chart_type"].map(lambda x: label_map[x])
    valid_df["label"] = valid_df["chart_type"].map(lambda x: label_map[x])

    train_dataset = datasets.ClassificationDataset(train_df, cfg.size, transform=get_train_transforms())
    val_dataset = datasets.ClassificationDataset(valid_df, cfg.size, transform=get_valid_transforms())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    ### Model
    class_num = cfg.class_num
    if cfg.model == 'swin_l_384':
        net = timm.create_model('swin_large_patch4_window12_384', pretrained=cfg.use_pretrained)
        net.head = torch.nn.Linear(1536, class_num, bias=True)
    elif cfg.model == "convnext_large_384_in22ft1k":
        net = timm.create_model('convnext_large_384_in22ft1k', pretrained=cfg.use_pretrained)
        net.head.fc = torch.nn.Linear(1536, class_num, bias=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    ### Loss
    criterion = nn.CrossEntropyLoss()

    ### Optimizer
    optimizer = Adafactor(net.parameters(), scale_parameter=False, relative_step=False, lr=Config.lr, weight_decay=1e-05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=int(len(train_df)//cfg.batch_size) * cfg.epochs)
    scaler = GradScaler()

    ### Train
    torch.backends.cudnn.benchmark = True
    def validataion():
        net.eval()
        losses = []
        preds = []
        targets = []
        with tqdm(val_dataloader, desc='Valid') as valid_pbar:
            for batch_idx, (images, labels) in enumerate(valid_pbar):
                with torch.no_grad():
                    y = net(images.to(device))
                    loss = criterion(y, labels.to(torch.long).to(device))
                y = F.softmax(y).cpu().numpy()
                labels = labels.cpu().numpy()

                _loss = float(loss.detach().cpu().numpy())
                if np.isnan(_loss) or np.isinf(_loss):
                    losses.append(1.0)
                else:
                    losses.append(_loss)
                preds.append(y)
                targets.append(labels)
                valid_pbar.set_postfix(OrderedDict(loss=np.array(losses).mean()))
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        return np.array(losses).mean(), preds, targets

    # Train Loop
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config.txt"), "w") as f:
        f.write(f"class_num: {Config.class_num}\n")
        f.write(f"size: {Config.size}\n")
        f.write(f"batch_size: {Config.batch_size}\n")
        f.write(f"epochs: {Config.epochs}\n")
        f.write(f"validation_step: {Config.validation_step}\n")
        f.write(f"model: {Config.model}\n")
        f.write(f"use_pretrained: {Config.use_pretrained}\n")
        f.write(f"num_warmup_steps: {Config.num_warmup_steps}\n")
        f.write(f"lr: {Config.lr}\n")
        f.write(f"patience_num: {Config.patience_num}\n")

    log = {
        "epoch": [],
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "acc_all": []
    }
    for k, v in label_map.items():
        log[f"acc_{k}"] = []

    best_loss = 100
    not_improve_counter = 0
    for epoch in tqdm(range(cfg.epochs), desc='Epoch'):
        if epoch==9:
            break

        net.train()
        losses = []
        with tqdm(train_dataloader, desc='Train') as train_pbar:
            for batch_idx, (images, labels) in enumerate(train_pbar):
                optimizer.zero_grad()
                with autocast():
                    y = net(images.to(device))
                    loss = criterion(y, labels.to(torch.long).to(device))

                _loss = float(loss.detach().cpu().numpy())
                if np.isnan(_loss) or np.isinf(_loss):
                    print("nan occur")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)

                #if scheduler._step_count < scheduler.total_steps:
                scheduler.step()
                scaler.update()

                if np.isnan(_loss) or np.isinf(_loss):
                    print("nan occur")
                    losses.append(1.0)
                else:
                    losses.append(_loss)
                train_pbar.set_postfix(OrderedDict(loss=np.array(losses).mean()))

                ### Validate in epoch
                if cfg.validation_step>0 and batch_idx>1 and batch_idx % cfg.validation_step == 0:
                    val_loss, preds, targets = validataion()
                    preds = preds.argmax(axis=1)
                    
                    acc_all = (preds==targets).sum() / len(targets)
                    acc_chart_types = {}
                    for k, v in label_map.items():
                        if len(targets[targets==v])==0:
                            acc_chart_types[f"acc_{k}"] = 0
                            continue
                        idx = targets==v
                        preds_ct = preds[idx]
                        targets_ct = targets[idx]
                        acc_ct = (preds_ct==targets_ct).sum() / len(targets_ct)
                        acc_chart_types[f"acc_{k}"] = acc_ct

                    if val_loss < best_loss:
                        print(f"Improve Best Score: {best_loss:.3f} -> {val_loss:.3f}")
                        best_loss = val_loss
                        torch.save(net.state_dict(), os.path.join(args.output, f"best.bin"))
                        not_improve_counter = 0
                    else:
                        not_improve_counter += 1
                        print(f"Not improve: {val_loss:.3f} (best={best_loss:.3f})/(not_improve_counter={not_improve_counter})")
                    
                    print(f"Acc = {acc_all}")
                    log["acc_all"] .append(acc_all)
                    for k, v in acc_chart_types.items():
                        print(f" {k} = {v}")
                        log[k].append(v)

                    log["epoch"].append(epoch)
                    log["step"].append(batch_idx)
                    log["train_loss"].append(float(np.array(losses).mean()))
                    log["val_loss"].append(val_loss)
                    log_df = pd.DataFrame.from_dict(log)
                    log_df.to_csv(os.path.join(args.output, f"log.csv"), index=False)

                    if not_improve_counter > Config.patience_num:
                        print("Early stopping!")
                        return
        
        ### Validate epoch end
        val_loss, preds, targets = validataion()
        preds = preds.argmax(axis=1)
        acc_all = (preds==targets).sum() / len(targets)
        acc_chart_types = {}
        for k, v in label_map.items():
            if len(targets[targets==v])==0:
                acc_chart_types[f"acc_{k}"] = 0
                continue
            idx = targets==v
            preds_ct = preds[idx]
            targets_ct = targets[idx]
            acc_ct = (preds_ct==targets_ct).sum() / len(targets_ct)
            acc_chart_types[f"acc_{k}"] = acc_ct

        if val_loss < best_loss:
            print(f"Improve Best Score: {best_loss:.3f} -> {val_loss:.3f}")
            best_loss = val_loss
            torch.save(net.state_dict(), os.path.join(args.output, f"best.bin"))
            not_improve_counter = 0
        else:
            not_improve_counter += 1
            print(f"Not improve: {val_loss:.3f} (best={best_loss:.3f})/(not_improve_counter={not_improve_counter})")
        
        print(f"Acc = {acc_all}")
        log["acc_all"].append(acc_all)
        for k, v in acc_chart_types.items():
            print(f" {k} = {v}")
            log[k].append(v)

        log["epoch"].append(epoch)
        log["step"].append(batch_idx)
        log["train_loss"].append(float(np.array(losses).mean()))
        log["val_loss"].append(val_loss)
        log_df = pd.DataFrame.from_dict(log)
        log_df.to_csv(os.path.join(args.output, f"log.csv"), index=False)

        if not_improve_counter > Config.patience_num:
            print("Early stopping!")
            return

if __name__ == '__main__':
    main()