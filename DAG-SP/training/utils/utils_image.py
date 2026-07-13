import torch
from tqdm import tqdm
import os
import torch.nn as nn
import yaml
import datetime
import numpy as np
import signal
import time
from utils.metrics import pixel_accuracy, meanIoU, weightedIoU, class_meanIoU, MetricMeter, PerClassMetricMeter
from data.utils.images_transforms import soft_to_hard_labels

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def train_one_epoch(
        train_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        device
):
    train_loss = 0
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Starting training epoch with {len(train_loader)} batches")
    print(f"Model type: {type(model)}")
    if isinstance(model, torch.nn.DataParallel):
        print(f"DataParallel devices: {model.device_ids}")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            
            images = images.float().contiguous()
            labels = labels.long().contiguous()
            
            # 强制将数据分配到正确的设备
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 直接使用模型，让DataParallel自动处理
            try:
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory in batch {batch_idx}! Clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    # 尝试减小批量大小
                    if images.size(0) > 1:
                        print("Trying smaller batch...")
                        # 手动分割批次
                        half = images.size(0) // 2
                        try:
                            # 处理第一半
                            out1 = model(images[:half])
                            loss1 = criterion(out1, labels[:half])
                            loss1.backward()
                            
                            # 处理第二半
                            out2 = model(images[half:])
                            loss2 = criterion(out2, labels[half:])
                            loss2.backward()
                            
                            # 平均梯度并更新
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad /= 2
                                    
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            train_loss += (loss1.item() + loss2.item()) / 2
                            print(f"Batch {batch_idx} completed with manual split!")
                        except Exception as split_e:
                            print(f"Manual split also failed: {split_e}")
                            optimizer.zero_grad()
                            continue
                    else:
                        optimizer.zero_grad()
                        continue
                else:
                    print(f"Runtime error in batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue
    
    # 显存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared after epoch")
    
    return train_loss / max(len(train_loader), 1)


def train_with_metrics(
        train_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        n_classes,
        ignore_index
):
    train_loss = 0
    train_iter = tqdm(train_loader)
    model.train()
    preds_for_iou = []
    labels_for_iou = []
    for (images, labels) in train_iter:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        loss = criterion(out, labels)

        preds = out.argmax(1)
        preds = preds.detach().cpu()
        if labels.dim() != preds.dim():
            labels = soft_to_hard_labels(labels, ignore_index)
        labels = labels.detach().cpu()
        preds_for_iou.append(preds)
        labels_for_iou.append(labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_iter.set_description(desc=f"train loss = {loss.item():.4f}")

    train_loss = train_loss/len(train_loader)
    preds_for_iou = torch.cat(preds_for_iou, dim=0)
    labels_for_iou = torch.cat(labels_for_iou, dim=0)
    global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    return train_loss, global_miou, global_classes_iou


def evaluate(
        val_loader,
        model,
        criterion,
        device
):
    val_loss = 0
    val_iter = tqdm(val_loader)
    model.eval()
    for (images, labels) in val_iter:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)

        val_loss += loss.item()
        val_iter.set_description(desc=f"val loss = {loss.item():.4f}")

    val_loss = val_loss/len(val_loader)
    return val_loss


def evaluate_with_metrics(
        val_loader,
        model,
        criterion,
        device,
        n_classes,
        ignore_index=255
):
    val_loss = 0
    val_iter = tqdm(val_loader)
    mIoU1 = MetricMeter()
    mIoU2 = MetricMeter()
    preds_for_iou = []
    labels_for_iou = []
    model.eval()
    
    with torch.no_grad():
        for (images, labels) in val_iter:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}")

    preds_for_iou = torch.cat(preds_for_iou, dim=0)
    labels_for_iou = torch.cat(labels_for_iou, dim=0)
    global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
    global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)

    val_loss = val_loss/len(val_loader)

    results = {
        "val_loss": val_loss,
        "mIoU1": mIoU1.avg,
        "mIoU2": mIoU2.avg,
        "global_miou": global_miou,
        "classes_mIoU": global_classes_iou
    }

    return results



def save_model(save_dict, cfg, save_dir, date, DATASET):
    date_str = date.strftime("%d-%m_%H-%M")
    ep = save_dict["epoch"]
    train_loss = save_dict["train_losses"][-1]
    train_global_miou = save_dict["train_global_miou"][-1]
    val_loss = save_dict["val_losses"][-1]
    val_global_miou = save_dict["val_global_miou"][-1]
    val_mIoU1 = save_dict["val_miou1"][-1]
    val_mIoU2 = save_dict["val_miou2"][-1]
    val_classes_mIoU = save_dict["val_classes_mIoU"][-1]
    train_classes_mIoU = save_dict["train_classes_mIoU"][-1]
    val_per_classes_mIoU = {v: val_classes_mIoU[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: val_classes_mIoU[k].item() for (k,v) in DATASET.classes.items()}
    train_per_classes_mIoU = {v: train_classes_mIoU[k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: train_classes_mIoU[k].item() for (k,v) in DATASET.classes.items()}

    save_name = f"{date_str}"
    if not os.path.exists(os.path.join(save_dir, save_name)):
        os.mkdir(os.path.join(save_dir, save_name))
    with open(os.path.join(save_dir, save_name, save_name + "_config.yaml"), "w") as file:
        yaml.dump(cfg, file, default_flow_style=False, sort_keys=False)

    ep_date = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    with open(os.path.join(save_dir, save_name, save_name + "_logs.txt"), "a") as f:
        f.write(ep_date)
        f.write(" -- ")
        f.write(f"Epoch {ep}")
        f.write(" -- ")
        f.write(f"train loss = {train_loss:.4f}, validation loss = {val_loss:.4f}, val mIoU = {val_global_miou:.4f}")
        f.write("\n")
        #if train_per_classes_mIoU is not None:
        #    f.write(f"Train per class mIoU: {train_per_classes_mIoU}\n")
        if val_per_classes_mIoU is not None:
            f.write(f"Val per class mIoU: {val_per_classes_mIoU}\n")

    torch.save(save_dict, os.path.join(save_dir, save_name, save_name + ".pth.tar"))
    return save_name