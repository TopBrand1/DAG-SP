import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import kornia.geometry as KG
import kornia.feature as KF

from utils.losses import TempConstLoss
from utils.torch_utils import conditional_no_grad
from utils.metrics import meanIoU, class_meanIoU, MetricMeter, PerClassMetricMeter
from data.utils.images_transforms import soft_to_hard_labels
from models.opt_flow import flowwarp, interpolate_flow
from models.video.mixing_layers import (
    STT3DMixingBlock,
    STT2DMixingBlock,
    Conv2DMixingBlock,
    Conv3DMixingBlock,
    IdMixingBlock)
from models.video.sim_layers import (
    ConstantSimBlock,
    AverageSimBlock,
    CossimSimBlock,
    ConvSimBlock,
    ConvSimBlockBase,
    ConvSimBlockv2,
    STT3DConvSimBlock,
    STT3DConvSimBlock2,
    STT3DCosSimBlock,
    ImprovedModulatedCrossAttention,
    SimplifiedModulatedAttentionSimBlock,
    MultiScaleLocalCrossAttention
)
from models.video.modules import Clip_PSP_Block, Clip_OCR_Block

def get_model(model_cfg, base_model, flow_model, n_classes):
    model_name = model_cfg.get("model_name", "base").lower()
    if model_name == "base":
        return ConsistencyWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "feat":
        return ConsistencyFeatureWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "dff":
        return DFFWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "dff2":
        return DFF2Wrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "netwarp":
        return NetWarpWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "tcb":
        return TCBWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "enhanced": 
        return EnhancedConsistencyWrapper(base_model, flow_model, n_classes, model_cfg)
    elif model_name == "tfc":
        return TFCWrapper(base_model, flow_model, n_classes, model_cfg)
    else:
        raise NotImplementedError()

def parse_mixing_ops(op_name, dim, n_layers, window_size=None, feat_mixing=False):
    op_name = op_name.lower()
    if op_name == "3dwmsa":
        if window_size is None:
            window_size = [2,7,7]
        return STT3DMixingBlock(dim, n_layers, window_size=window_size, shift_size=[0,3,3], n_heads=1, hidden_dim=4*dim)
    elif op_name == "2dwmsa":
        if window_size is None:
            window_size = [9,9]
        return STT2DMixingBlock(dim*2, n_layers, window_size=window_size, shift_size=[3,3], n_heads=1, hidden_dim=8*dim)
    elif op_name == "2d_conv":
        return Conv2DMixingBlock(dim*2, n_layers, hidden_dim=8*dim)
    elif op_name == "3d_conv":
        return Conv3DMixingBlock(dim, n_layers, hidden_dim=4*dim)
    elif op_name == "id":
        return IdMixingBlock()
    else:
        print(f"Mixing op {op_name} is not implemented")
        raise NotImplementedError()
    
def parse_sim_ops(op_name, embed_dim):
    op_name = op_name.lower()
    if op_name == "cossim":
        return CossimSimBlock()
    elif op_name == "conv":
        return ConvSimBlock(embed_dim)
    elif op_name == "tfc":
        return TFCCosSimBlock()
    elif op_name == "convbase":
        return ConvSimBlockBase(embed_dim)
    elif op_name == "conv2":
        return ConvSimBlockv2(embed_dim)
    elif op_name == "constant":
        return ConstantSimBlock(0.)
    elif op_name == "avg":
        return AverageSimBlock(0.5)
    elif op_name == "3dwmsaconv":
        return STT3DConvSimBlock(embed_dim)
    elif op_name == "3dwmsaconv2":
        return STT3DConvSimBlock2(embed_dim)
    elif op_name == "3dwmsacos":
        return STT3DCosSimBlock(embed_dim)
    elif op_name == "reuse":
        return ConstantSimBlock(1.)
    elif op_name == "efficient_attention": 
        return ImprovedModulatedCrossAttention(embed_dim)
    elif op_name == "simplified_attention":
        return SimplifiedModulatedAttentionSimBlock(embed_dim)
    elif op_name == "multi_scale_local_cross_attention":  
        return MultiScaleLocalCrossAttention(embed_dim)
    else:
        print(f"Similarity op {op_name} is not implemented")
        raise NotImplementedError()


def get_matching_kpts(lafs1, lafs2, idxs):
    src_pts = KF.get_laf_center(lafs1).view(-1, 2)[idxs[:, 0]]
    dst_pts = KF.get_laf_center(lafs2).view(-1, 2)[idxs[:, 1]]
    return src_pts, dst_pts

class TFCCosSimBlock(nn.Module):
    """
    TFC-style parameter-free cosine similarity.

    Given two low-level feature maps F_prev and F_cur,
    compute a pixel-wise similarity map:
        sim = <normalize(F_prev), normalize(F_cur)>
    Then clamp sim into [0, 1] and upsample it to the output size.

    This implements the core TFC interpolation weight under our propagation framework.
    """
    def __init__(self, eps=1e-6, clamp=True):
        super().__init__()
        self.eps = eps
        self.clamp = clamp

    def forward(self, last_feat, feat, size=None):
        # last_feat, feat: [B, C, H, W]
        if last_feat.shape[-2:] != feat.shape[-2:]:
            last_feat = F.interpolate(
                last_feat,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        last_feat = F.normalize(last_feat, p=2, dim=1, eps=self.eps)
        feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

        sim = torch.sum(last_feat * feat, dim=1, keepdim=True)

        if self.clamp:
            sim = torch.clamp(sim, min=0.0, max=1.0)

        if size is not None and sim.shape[-2:] != size:
            sim = F.interpolate(
                sim,
                size=size,
                mode="bilinear",
                align_corners=False
            )

        return sim
    
class DiskFeatureRegistrator(nn.Module):
    def __init__(self, low_scale_homo):
        super().__init__()
        self.num_features = 2048
        self.disk = KF.DISK.from_pretrained("depth")
        self.lg_matcher = KF.LightGlueMatcher("disk").eval()
        self.ransac = KG.RANSAC(model_type="homography", inl_th=2.5)

        self.low_scale_homo = low_scale_homo

    def forward(self, last_frame, frame):
        if self.low_scale_homo:
            last_frame = nn.functional.interpolate(last_frame, scale_factor=1/4, mode="bilinear", antialias=True)
            frame = nn.functional.interpolate(frame, scale_factor=1/4, mode="bilinear", antialias=True)
        hw1 = torch.tensor(last_frame.shape[2:], device=frame.device)
        hw2 = torch.tensor(frame.shape[2:], device=frame.device)
        with torch.inference_mode():
            inp = torch.cat([last_frame, frame], dim=0)
            features1, features2 = self.disk(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=frame.device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=frame.device))

            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        if len(idxs) >= 4:
            src_pts, dst_pts = get_matching_kpts(lafs1, lafs2, idxs)
            homo, mask = self.ransac(src_pts, dst_pts)
        else:
            homo = torch.tensor([[1.,0,0], [0,1.,0], [0,0,1.]], device=frame.device)
            print("not computed")
        return homo.unsqueeze(0)

    def batch_compute(self, last_frames, frames):
        homos = []
        for k in range(len(frames)):
            h = self(last_frames[k].unsqueeze(0), frames[k].unsqueeze(0)).detach()
            homos.append(h)
        return homos
    
#这个类总是返回单位单应性矩阵，也就是说假设相邻帧之间没有运动，或者我们不对相邻帧进行配准
class IdentityRegistrator(nn.Module):
    def __init__(self, low_scale_homo):
        super().__init__()
    def forward(self, last_frame, frame):
        homo = torch.tensor([[1.,0,0], [0,1.,0], [0,0,1.]], device=frame.device)
        return homo.unsqueeze(0)
    def batch_compute(self, last_frames, frames):
        homos = []
        for k in range(len(frames)):
            h = self(last_frames[k].unsqueeze(0), frames[k].unsqueeze(0)).detach()
            homos.append(h)
        return homos


def warp_batch(homos, last_out, scale=None, interp_mode="bilinear"):
    warped_list = []
    for k in range(len(last_out)):
        homo = homos[k]
        if scale is not None:
            down = torch.tensor([[[scale,0,0], [0,scale,0], [0,0,1]]], device=homo.device)
            up = torch.tensor([[[1/scale,0,0], [0,1/scale,0], [0,0,1]]], device=homo.device)
            homo = down @ homo @ up
        warped_list.append(KG.warp_perspective(last_out[k].unsqueeze(0), homo, last_out[k].shape[-2:], mode=interp_mode))
    last_out_warped = torch.cat(warped_list, dim=0) 
    return last_out_warped

def warp(homo, last_out, scale=None, interp_mode="bilinear"):
    if scale is not None:
        down = torch.tensor([[[scale,0,0], [0,scale,0], [0,0,1]]], device=homo.device)
        up = torch.tensor([[[1/scale,0,0], [0,1/scale,0], [0,0,1]]], device=homo.device)
        homo = down @ homo @ up
    last_out_warped = KG.warp_perspective(last_out, homo, last_out.shape[-2:], mode=interp_mode)
    return last_out_warped
    
class ConsistencyWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.pred_mixing_op = model_cfg["pred_mixing_op"]
        self.sim_op = model_cfg["sim_op"]
        self.n_mixing_layers = model_cfg.get("n_mixing_layers", 1)
        self.window_size = model_cfg.get("window_size", None)
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.kd_consistent_labels = model_cfg.get("kd_consistent_labels", True)
        self.last_out_loss_lb = model_cfg.get("last_out_loss_lb", 0) # For KD only

        self.upsample_before_mixing = model_cfg.get("upsample_before_mixing", True)
        self.low_scale_homo = model_cfg.get("low_scale_homo", False)

        self.features_interp_mode = model_cfg.get("features_interp_mode", "bilinear")
        self.k_registrator_model = model_cfg.get("k_registrator_model", None)
        if self.k_registrator_model == "disk":
            self.registrator = DiskFeatureRegistrator(self.low_scale_homo)
        elif self.k_registrator_model == "identity":
            self.registrator = IdentityRegistrator(self.low_scale_homo)
        else:
            self.registrator = None

        self.pred_mixing = parse_mixing_ops(self.pred_mixing_op, self.n_classes, self.n_mixing_layers, self.window_size)
        try:
            encoder_size = self.base_model.model.config.hidden_sizes[0]
        except:
            encoder_size = 96
        self.sim = parse_sim_ops(self.sim_op, encoder_size)

# forward步骤
# a. 如果没有提供homos，则使用registrator（这里是IdentityRegistrator）计算相邻帧的单应性矩阵（恒等矩阵）。
# b. 提取上一帧和当前帧的特征和输出（logits）。
# c. 如果设置了upsample_before_mixing，则将输出上采样到原始帧大小。
# d. 使用计算出的单应性矩阵对上一帧的输出和特征进行warp（由于是恒等矩阵，warp后的结果和原始结果相同）。
# e. 将warped的上一帧输出和当前帧输出通过mixing块（这里是不做操作的IdMixingBlock）。
# f. 使用相似度模块计算相似度图，然后利用相似度图将warped的上一帧输出和当前帧输出进行融合。
    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        # 使用IdentityRegistrator获取单应性矩阵（单位矩阵）
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        # 提取特征和预测
        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)
            out = self.base_model.decoder(feats)
            last_featmap, featmap = last_feats[0], feats[0]

            if self.upsample_before_mixing and out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
        # 使用单位矩阵进行warp（实际上不做变换）
            last_out_warped = warp_batch(homos, last_out, scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4)
            last_featmap_warped = warp_batch(homos, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)

            # Mixing 使用IdMixingBlock（不做任何混合）
            last_out_warped, out = self.pred_mixing(last_out_warped, out)
            # Similarity map 使用ConvSimBlock计算相似度图
            sim = self.sim(last_featmap_warped, featmap, size=out.shape[-2:])
            # Fusion based on similarity map 基于相似度进行特征融合
            out = sim*last_out_warped + (1-sim)*out

        if return_last_out:
            if return_sim:
                return out, last_out, sim
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, homos=homos, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        
        # 创建MetricMeter实例
        from utils.metrics import MetricMeter
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        
        # 存储每个批次的预测和标签用于计算全局统计
        all_preds = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, (frames, adj_frames, labels, homos) in enumerate(val_iter):
                adj_frames = [adj_f.to(device) for adj_f in adj_frames]
                frames = frames.to(device)
                labels = labels.to(device)

                # Check if precomputed homos
                if homos.dim() < 3:
                    homos = None
                else:
                    homos = homos.to(device)
                
                last_frames = adj_frames[0]
                out, last_out = self.forward(frames, last_frames, homos=homos, eval=True, return_last_out=True)

                if out.shape[-2:] != frames.shape[-2:]:
                    out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                    last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

                loss = criterion(out, labels)
                if self.const_loss_lb > 0:
                    const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                    loss += const

                # 计算预测
                preds = out.argmax(1)
                preds = preds.detach().cpu()
                
                # 处理标签维度
                if labels.dim() != preds.dim():
                    from data.utils.images_transforms import soft_to_hard_labels
                    labels = soft_to_hard_labels(labels, ignore_index)
                labels = labels.detach().cpu()
                
                # 存储当前批次的预测和标签
                all_preds.append(preds)
                all_labels.append(labels)
                
                # 计算当前批次的mIoU并更新MetricMeters
                if len(preds) > 0:
                    from utils.metrics import meanIoU
                    batch_miou = meanIoU(preds, labels, n_classes, ignore_index=ignore_index)
                    mIoU1.update(batch_miou)
                    mIoU2.update(batch_miou)
                
                val_loss += loss.item()
                const_value = const.item() if const is not None else 0
                const_loss += const_value
                val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")
        
        # 计算全局统计
        val_loss = val_loss / len(val_loader)
        const_loss = const_loss / len(val_loader)
        
        # 计算全局mIoU和每个类别的IoU
        global_miou = 0.0
        classes_iou = torch.zeros(n_classes)
        
        if all_preds:
            # 合并所有批次的预测和标签
            all_preds_tensor = torch.cat(all_preds, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            
            # 计算全局mIoU
            from utils.metrics import meanIoU, class_meanIoU
            global_miou = meanIoU(all_preds_tensor, all_labels_tensor, n_classes, ignore_index=ignore_index)
            classes_iou = class_meanIoU(all_preds_tensor, all_labels_tensor, n_classes, ignore_index=ignore_index)
        
        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": classes_iou
        }

        return results
    
# 对于知识蒸馏（KD）训练，该方法使用教师模型的输出的相邻帧的标签来训练学生模型。
# 由于k_registrator_model是identity，所以单应性矩阵总是单位矩阵，因此warp操作实际上没有改变上一帧的输出和特征。
# 由于pred_mixing_op是id，所以mixing块不改变两个输出。
# 相似度图是通过卷积网络计算的，所以模型会学习如何融合两个输出。
    def kd_train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, adj_labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            adj_labels = [adj_l.to(device) for adj_l in adj_labels]
            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            last_labels = adj_labels[0]
            out, last_out, sim = self.forward(frames, last_frames, homos=homos, eval=False, return_last_out=True, return_sim=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                sim = nn.functional.interpolate(sim, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            if self.kd_consistent_labels:
                last_labels, labels = self.kd_get_consistent_logits(last_frames, frames, last_labels, labels)

            loss = criterion(out, labels) + self.last_out_loss_lb * criterion(last_out, last_labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def kd_get_consistent_logits(self, last_frames, frames, last_labels, labels):
        with torch.no_grad():
            self.flow_model.eval()
            _, flow_last2current = self.flow_model(frames, last_frames, iters=15, test_mode=True)
            _, flow_current2last = self.flow_model(last_frames, frames, iters=15, test_mode=True)
        occlusion_mask = (((flow_last2current + flow_current2last)**2).sum(1) > 0.01*((flow_last2current**2).sum(1) + (flow_current2last**2).sum(1)) + 0.5).unsqueeze(1)

        warped_last_labels = flowwarp(last_labels, flow_last2current)
        warped_labels = flowwarp(labels, flow_current2last)

        last_labels_consit = (last_labels + warped_labels*(~occlusion_mask)) / (2*(~occlusion_mask) + 1*occlusion_mask)
        labels_consit = (labels + warped_last_labels*(~occlusion_mask)) / (2*(~occlusion_mask) + 1*occlusion_mask)
        return last_labels_consit, labels_consit
    
    def infer_frame(self, frame, homo=None, last_frame=None, last_out=None, last_feats=None, return_attention=False):
        attention_info = None

        with torch.no_grad():
            feats = self.base_model.encoder(frame)
            out = self.base_model.decoder(feats)

        if last_out is not None:
            last_featmap, featmap = last_feats[0], feats[0]

            if homo is None:
                homo = self.registrator(last_frame, frame)

            with torch.no_grad():
                if self.upsample_before_mixing and out.shape[-2:] != frame.shape[-2:]:
                    out = F.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
                    last_out = F.interpolate(last_out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)

                last_out_warped = warp(homo, last_out, scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4)
                last_featmap_warped = warp(homo, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)

                last_out_warped, out = self.pred_mixing(last_out_warped, out)

                if return_attention:
                    try:
                        sim_result = self.sim(
                            last_featmap_warped,
                            featmap,
                            size=out.shape[-2:],
                            return_attention=True
                        )

                        if isinstance(sim_result, tuple):
                            sim, attention_info = sim_result
                        else:
                            sim = sim_result
                            attention_info = {
                                "method": self.sim.__class__.__name__,
                                "sim": sim
                            }

                    except TypeError as e:
                        # Some similarity modules, e.g., ConvSimBlock / old CossimSimBlock,
                        # do not support return_attention. Fall back to normal forward.
                        if "return_attention" not in str(e):
                            raise e

                        sim = self.sim(
                            last_featmap_warped,
                            featmap,
                            size=out.shape[-2:]
                        )

                        attention_info = {
                            "method": self.sim.__class__.__name__,
                            "sim": sim
                        }
                else:
                    sim = self.sim(
                        last_featmap_warped,
                        featmap,
                        size=out.shape[-2:]
                    )

                out = sim * last_out_warped + (1 - sim) * out

        if return_attention:
            return out, feats, attention_info

        return out, feats
    
    def infer_video(self, frames, homos, device, return_attention=False):
        self.eval()
        preds = []
        attention_records = []

        last_frame = None
        last_out = None
        last_feats = None

        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)

            homo = homos[i]
            if homo is not None:
                homo = homo.to(device)

            if return_attention:
                out, feats, attention_info = self.infer_frame(
                    frame,
                    homo,
                    last_frame,
                    last_out,
                    last_feats,
                    return_attention=True
                )
                attention_records.append(attention_info)
            else:
                out, feats = self.infer_frame(
                    frame,
                    homo,
                    last_frame,
                    last_out,
                    last_feats
                )

            last_feats = feats
            last_out = out
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = F.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)

            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)

        if return_attention:
            return preds, attention_records

        return preds
    
class TFCWrapper(ConsistencyWrapper):
    """
    TFC-style baseline under the current DAG-SP/SSP propagation framework.

    Difference from the original TFC:
    - Original TFC interpolates decoder features before the class prediction head.
    - This wrapper interpolates segmentation logits/predictions, consistent with
      the existing ConsistencyWrapper/DAG-SP evaluation pipeline.
    - Sparse MSDA is not included here, because the current base_model.decoder
      does not expose the intermediate decoder feature z before classification.

    This is intended as a controlled baseline:
        parameter-free cosine similarity vs. learnable ConvSim / DAG-SP attention.
    """
    def __init__(self, base_model, flow_model, n_classes, model_cfg):
        super().__init__(base_model, flow_model, n_classes, model_cfg)

        self.sim = TFCCosSimBlock(
            eps=model_cfg.get("tfc_eps", 1e-6),
            clamp=model_cfg.get("tfc_clamp", True)
        )

        # TFC baseline should not use a learnable prediction mixing module.
        self.pred_mixing = IdMixingBlock()

        # TFC paper favors low-level feature correlation.
        # feats[0] usually corresponds to 1/4 resolution;
        # feats[1] usually corresponds to 1/8 resolution.
        self.tfc_feat_level = model_cfg.get("tfc_feat_level", 1)

        # Whether to use homography/global registration.
        # For strict TFC-style baseline, use identity.
        # For fair comparison under our framework, use disk/precomputed homos.
        self.use_warped_feature_for_tfc = model_cfg.get("use_warped_feature_for_tfc", True)

    def _get_feature_scale(self, level):
        """
        Estimate homography scaling for feature level.
        Assumption:
            level 0 -> 1/4 resolution
            level 1 -> 1/8 resolution
            level 2 -> 1/16 resolution
            level 3 -> 1/32 resolution
        """
        if self.low_scale_homo:
            return None
        return 1.0 / (4 * (2 ** level))

    def forward(self, frames, last_frames, homos=None, eval=False,
                return_last_out=False, return_sim=False):

        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)

            feats = self.base_model.encoder(frames)
            out = self.base_model.decoder(feats)

            if self.upsample_before_mixing and out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(
                    out,
                    size=frames.shape[-2:],
                    mode=self.upsampling,
                    align_corners=False
                )
                last_out = F.interpolate(
                    last_out,
                    size=frames.shape[-2:],
                    mode=self.upsampling,
                    align_corners=False
                )

            # Select low-level features for TFC cosine similarity.
            level = min(self.tfc_feat_level, len(feats) - 1)
            last_featmap = last_feats[level]
            featmap = feats[level]

            # Warp previous prediction/logit.
            last_out_warped = warp_batch(
                homos,
                last_out,
                scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4
            )

            # TFC-style: optionally warp previous low-level feature before cosine similarity.
            # If k_registrator_model="identity", this degenerates to original unaligned cosine matching.
            if self.use_warped_feature_for_tfc:
                last_featmap_for_sim = warp_batch(
                    homos,
                    last_featmap,
                    scale=self._get_feature_scale(level),
                    interp_mode=self.features_interp_mode
                )
            else:
                last_featmap_for_sim = last_featmap

            # No learnable prediction mixing.
            last_out_warped, out = self.pred_mixing(last_out_warped, out)

            # Parameter-free cosine similarity.
            sim = self.sim(last_featmap_for_sim, featmap, size=out.shape[-2:])

            # TFC-style linear interpolation.
            out = sim * last_out_warped + (1.0 - sim) * out

        if return_last_out:
            if return_sim:
                return out, last_out, sim
            return out, last_out

        return out

    def infer_frame(
            self,
            frame,
            homo=None,
            last_frame=None,
            last_out=None,
            last_feats=None,
            return_attention=False
    ):
        """
        TFC-style inference for one frame.

        return_attention=False:
            return out, feats

        return_attention=True:
            return out, feats, attention_info

        For TFC, attention_info only contains similarity map.
        It does not contain channel attention, because TFC has no channel-attention module.
        """
        attention_info = None

        with torch.no_grad():
            feats = self.base_model.encoder(frame)
            out = self.base_model.decoder(feats)

        if last_out is not None:
            if homo is None:
                homo = self.registrator(last_frame, frame)

            with torch.no_grad():
                if self.upsample_before_mixing and out.shape[-2:] != frame.shape[-2:]:
                    out = F.interpolate(
                        out,
                        size=frame.shape[-2:],
                        mode=self.upsampling,
                        align_corners=False
                    )
                    last_out = F.interpolate(
                        last_out,
                        size=frame.shape[-2:],
                        mode=self.upsampling,
                        align_corners=False
                    )

                # Select low-level feature for TFC cosine similarity.
                level = min(self.tfc_feat_level, len(feats) - 1)
                last_featmap = last_feats[level]
                featmap = feats[level]

                # Warp previous prediction/logit.
                last_out_warped = warp(
                    homo,
                    last_out,
                    scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1 / 4
                )

                # Warp previous feature if using +H setting.
                if self.use_warped_feature_for_tfc:
                    last_featmap_for_sim = warp(
                        homo,
                        last_featmap,
                        scale=self._get_feature_scale(level),
                        interp_mode=self.features_interp_mode
                    )
                else:
                    last_featmap_for_sim = last_featmap

                # TFC uses no learnable prediction mixing.
                last_out_warped, out = self.pred_mixing(last_out_warped, out)

                # Compute TFC-style cosine similarity map.
                sim = self.sim(
                    last_featmap_for_sim,
                    featmap,
                    size=out.shape[-2:]
                )

                if return_attention:
                    attention_info = {
                        "method": "tfc_cosine_similarity",
                        "sim": sim
                    }

                out = sim * last_out_warped + (1.0 - sim) * out

        if return_attention:
            return out, feats, attention_info

        return out, feats
    
class EnhancedTemporalConsistencyLoss(nn.Module):
    """增强的时序一致性损失"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
            super(EnhancedTemporalConsistencyLoss, self).__init__()
            self.alpha = alpha  # 预测一致性权重
            self.beta = beta    # 特征平滑性权重
            self.gamma = gamma  # 多尺度一致性权重
        
    def compute_prediction_consistency(self, pred_current, pred_prev):
        """预测一致性约束"""
        return F.l1_loss(pred_current, pred_prev, reduction='mean')
    
    def compute_feature_smoothness(self, feat_current, feat_prev):
        """特征平滑性约束"""
        # 确保输入是张量
        if isinstance(feat_current, list):
            feat_current = feat_current[0]
        if isinstance(feat_prev, list):
            feat_prev = feat_prev[0]
            
        # 计算特征梯度差异
        feat_current_grad_x = torch.abs(feat_current[:, :, :, 1:] - feat_current[:, :, :, :-1])
        feat_current_grad_y = torch.abs(feat_current[:, :, 1:, :] - feat_current[:, :, :-1, :])
        feat_prev_grad_x = torch.abs(feat_prev[:, :, :, 1:] - feat_prev[:, :, :, :-1])
        feat_prev_grad_y = torch.abs(feat_prev[:, :, 1:, :] - feat_prev[:, :, :-1, :])
        
        # 对齐梯度尺寸
        min_h = min(feat_current_grad_y.shape[2], feat_prev_grad_y.shape[2])
        min_w = min(feat_current_grad_x.shape[3], feat_prev_grad_x.shape[3])
        
        grad_diff_x = torch.abs(feat_current_grad_x[:, :, :, :min_w] - feat_prev_grad_x[:, :, :, :min_w])
        grad_diff_y = torch.abs(feat_current_grad_y[:, :, :min_h, :] - feat_prev_grad_y[:, :, :min_h, :])
        
        smoothness_loss = (grad_diff_x.mean() + grad_diff_y.mean()) / 2
        return smoothness_loss
    
    def compute_multi_scale_consistency(self, feat_pyramid_current, feat_pyramid_prev):
        """多尺度特征一致性"""
        if feat_pyramid_current is None or feat_pyramid_prev is None:
            return 0.0
            
        multi_scale_loss = 0.0
        num_scales = min(len(feat_pyramid_current), len(feat_pyramid_prev))
        
        for i in range(num_scales):
            feat_curr = feat_pyramid_current[i]
            feat_prev = feat_pyramid_prev[i]
            
            # 确保特征尺寸匹配
            if feat_curr.shape != feat_prev.shape:
                if feat_curr.shape[-2:] != feat_prev.shape[-2:]:
                    feat_prev = F.interpolate(feat_prev, size=feat_curr.shape[-2:], mode='bilinear', align_corners=False)
            
            scale_loss = F.l1_loss(feat_curr, feat_prev, reduction='mean')
            multi_scale_loss += scale_loss
        
        return multi_scale_loss / num_scales if num_scales > 0 else 0.0
    
    def forward(self, pred_current, pred_prev, feat_current, feat_prev, 
                feat_pyramid_current=None, feat_pyramid_prev=None):
        # 各项损失分量
        pred_consistency_loss = self.compute_prediction_consistency(pred_current, pred_prev)
        feature_smoothness_loss = self.compute_feature_smoothness(feat_current, feat_prev)
        
        # 总损失
        total_loss = (self.alpha * pred_consistency_loss + 
                     self.beta * feature_smoothness_loss)
        
        # 如果提供了多尺度特征，添加多尺度一致性损失
        if feat_pyramid_current is not None and feat_pyramid_prev is not None:
            multi_scale_loss = self.compute_multi_scale_consistency(
                feat_pyramid_current, feat_pyramid_prev
            )
            total_loss += self.gamma * multi_scale_loss
        
        return total_loss


    
class EnhancedConsistencyWrapper(ConsistencyWrapper):
    """增强的一致性包装器，使用调制交叉注意力和轻量级时序约束"""
    def __init__(self, base_model, flow_model, n_classes, model_cfg):
            super().__init__(base_model, flow_model, n_classes, model_cfg)
            
            # 重新初始化相似度模块
            try:
                encoder_size = self.base_model.model.config.hidden_sizes[0]
            except:
                encoder_size = 96
                
            # 根据配置选择相似度模块
            sim_op = model_cfg.get("sim_op", "conv")
            if sim_op == "efficient_attention":
                self.sim = ImprovedModulatedCrossAttention(encoder_size)
            elif sim_op == "multi_scale_local_cross_attention":
                self.sim = MultiScaleLocalCrossAttention(encoder_size)
            elif sim_op == "improved_attention":
                self.sim = ImprovedModulatedCrossAttention(
                    embed_dim=encoder_size,
                    num_heads=model_cfg.get("num_heads", 4),
                    dropout=model_cfg.get("dropout", 0.1)
                )
            elif sim_op == "simplified_attention":
                self.sim = SimplifiedModulatedAttentionSimBlock(encoder_size)
            else:
                self.sim = parse_sim_ops(sim_op, encoder_size)
            
            # 增强的时序一致性损失
            if model_cfg.get("use_enhanced_temporal_loss", True):
                self.const_loss = EnhancedTemporalConsistencyLoss(
                    alpha=model_cfg.get("alpha", 0.2),
                    beta=model_cfg.get("beta", 0.05),
                    gamma=model_cfg.get("gamma", 0.02)
                )
            else:
                from utils.losses import TempConstLoss
                self.const_loss = TempConstLoss()
            
            # 是否使用多尺度特征
            self.use_multi_scale_features = model_cfg.get("use_multi_scale_features", True)
        
    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)
            out = self.base_model.decoder(feats)
            
            # 获取多尺度特征（前3层）
            if self.use_multi_scale_features:
                last_featmap_multi = last_feats[:3]  # 取前3层特征
                featmap_multi = feats[:3]
                last_featmap = last_featmap_multi[0]  # 主要特征用于相似度计算
                featmap = featmap_multi[0]
            else:
                last_featmap, featmap = last_feats[0], feats[0]
                last_featmap_multi = [last_featmap]
                featmap_multi = [featmap]

            if self.upsample_before_mixing and out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                
            # Warp操作
            last_out_warped = warp_batch(homos, last_out, scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4)
            last_featmap_warped = warp_batch(homos, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)
            
            if self.use_multi_scale_features:
                last_featmap_multi_warped = []
                for i, feat in enumerate(last_featmap_multi):
                    scale_factor = 1.0 / (4 * (2 ** i)) if not self.low_scale_homo else None
                    warped = warp_batch(homos, feat, scale=scale_factor, interp_mode=self.features_interp_mode)
                    last_featmap_multi_warped.append(warped)


            last_out_warped, out = self.pred_mixing(last_out_warped, out)

            sim = self.sim(last_featmap_warped, featmap, size=out.shape[-2:])
            

            out = sim * last_out_warped + (1 - sim) * out

        if return_last_out:
            if return_sim:
                if self.use_multi_scale_features:
                    return out, last_out, sim, featmap_multi, last_featmap_multi_warped
                else:
                    return out, last_out, sim, [featmap], [last_featmap_warped]
            return out, last_out
        return out

    def train_one_epoch(self, train_loader, optimizer, criterion, scheduler, device):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            
            if self.use_multi_scale_features:
                out, last_out, sim, featmap_multi, last_featmap_multi = self.forward(
                    frames, last_frames, homos=homos, eval=False, 
                    return_last_out=True, return_sim=True
                )
            else:
                out, last_out, sim, featmap_multi, last_featmap_multi = self.forward(
                    frames, last_frames, homos=homos, eval=False,
                    return_last_out=True, return_sim=True
                )
                featmap_multi = [featmap_multi]
                last_featmap_multi = [last_featmap_multi]

            if out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(
                    out, last_out, 
                    featmap_multi[0], last_featmap_multi[0],
                    featmap_multi, last_featmap_multi
                )
                loss += const

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
            
        return train_loss/len(train_loader), const_loss/len(train_loader)
    
    def kd_train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, adj_labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            adj_labels = [adj_l.to(device) for adj_l in adj_labels]
            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)

            last_frames = adj_frames[0]
            last_labels = adj_labels[0]
            
            if self.use_multi_scale_features:
                out, last_out, sim, featmap_multi, last_featmap_multi = self.forward(
                    frames, last_frames, homos=homos, eval=False, 
                    return_last_out=True, return_sim=True
                )
            else:
                out, last_out, sim, featmap_multi, last_featmap_multi = self.forward(
                    frames, last_frames, homos=homos, eval=False,
                    return_last_out=True, return_sim=True
                )
                featmap_multi = [featmap_multi]
                last_featmap_multi = [last_featmap_multi]

            if out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                sim = F.interpolate(sim, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            if self.kd_consistent_labels:
                last_labels, labels = self.kd_get_consistent_logits(last_frames, frames, last_labels, labels)

            loss = criterion(out, labels) + self.last_out_loss_lb * criterion(last_out, last_labels)
            
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(
                    out, last_out, 
                    featmap_multi[0], last_featmap_multi[0],
                    featmap_multi, last_featmap_multi
                )
                loss += const

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        
        return train_loss/len(train_loader), const_loss/len(train_loader)

    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            # Check if precomputed homos
            if homos.dim() < 3:
                homos = None
            else:
                homos = homos.to(device)
            
            last_frames = adj_frames[0]
            if self.use_multi_scale_features:
                out, last_out, sim, featmap_multi, last_featmap_multi = self.forward(
                    frames, last_frames, homos=homos, eval=True, return_last_out=True, return_sim=True
                )
                feat_current = featmap_multi[0]
                feat_prev = last_featmap_multi[0]
            else:
                out, last_out, sim, featmap, last_featmap = self.forward(
                    frames, last_frames, homos=homos, eval=True, return_last_out=True, return_sim=True
                )
                feat_current = featmap
                feat_prev = last_featmap

            if out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                if isinstance(self.const_loss, EnhancedTemporalConsistencyLoss):
                    const = self.const_loss_lb * self.const_loss(
                        out, last_out, feat_current, feat_prev, None, None
                    )
                else:
                    const = self.const_loss_lb * self.const_loss(out, last_out, feat_current, feat_prev)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results

    
class MultiScaleConsistencyWrapper(ConsistencyWrapper):
    def __init__(self, base_model, flow_model, n_classes, model_cfg):
        super().__init__(base_model, flow_model, n_classes, model_cfg)
        
        try:
            encoder_size = self.base_model.model.config.hidden_sizes[0]
        except:
            encoder_size = 96
            
        self.sim = MultiScaleLocalCrossAttention(encoder_size)
        
        self.use_multi_scale_fusion = model_cfg.get("use_multi_scale_fusion", False)
        if self.use_multi_scale_fusion:
            self.multi_scale_fusion = nn.Sequential(
                nn.Conv2d(n_classes * 2, n_classes, 3, padding=1),
                nn.BatchNorm2d(n_classes),
                nn.ReLU(),
                nn.Conv2d(n_classes, n_classes, 1)
            )
    
    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)
            out = self.base_model.decoder(feats)
            last_featmap, featmap = last_feats[0], feats[0]

            if self.upsample_before_mixing and out.shape[-2:] != frames.shape[-2:]:
                out = F.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = F.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                
            last_out_warped = warp_batch(homos, last_out, scale=None if (self.upsample_before_mixing or self.low_scale_homo) else 1/4)
            last_featmap_warped = warp_batch(homos, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)

            if self.use_multi_scale_fusion:

                multi_scale_feats = []
                for feat in feats[1:3]: 
                    if feat.shape[-2:] != out.shape[-2:]:
                        feat = F.interpolate(feat, size=out.shape[-2:], mode="bilinear", align_corners=False)
                    multi_scale_feats.append(feat)
                
                if multi_scale_feats:
                    combined_feat = torch.cat([featmap] + multi_scale_feats, dim=1)
                    featmap = self.multi_scale_fusion(combined_feat)
            

            last_out_warped, out = self.pred_mixing(last_out_warped, out)
            
            sim = self.sim(last_featmap_warped, featmap, size=out.shape[-2:])
            

            out = sim * last_out_warped + (1 - sim) * out

        if return_last_out:
            if return_sim:
                return out, last_out, sim, featmap, last_featmap
            return out, last_out
        return out
    
class ConsistencyFeatureWrapper(ConsistencyWrapper):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__(base_model, flow_model, n_classes, model_cfg)
        embed_dim = 512
        self.pred_mixing = parse_mixing_ops(self.pred_mixing_op, embed_dim, self.n_mixing_layers, self.window_size, feat_mixing=True)


    def forward(self, frames, last_frames, homos=None, eval=False, return_last_out=False, return_sim=False):
        if homos is None:
            homos = self.registrator.batch_compute(last_frames, frames)

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_decoderfeats = self.base_model.decoder_lastfeat(last_feats)
            feats = self.base_model.encoder(frames)
            decoderfeats = self.base_model.decoder_lastfeat(feats)
            last_featmap, featmap = last_feats[0], feats[0]

            last_decoderfeats_warped = warp_batch(homos, last_decoderfeats, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)
            last_featmap_warped = warp_batch(homos, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)

            # Mixing
            last_decoderfeats_warped, decoderfeats = self.pred_mixing(last_decoderfeats_warped, decoderfeats)
            # Similarity map
            sim = self.sim(last_featmap_warped, featmap, size=decoderfeats.shape[-2:])
            # Fusion based on similarity map
            #decoderfeats = sim*last_decoderfeats_warped + (1-sim)*decoderfeats
            # Final classifying layer
            out = self.base_model.forward_classifier(decoderfeats)
            last_out_warped = self.base_model.forward_classifier(last_decoderfeats_warped)
            # Fusion on predictions
            out = sim*last_out_warped + (1-sim)*out

        if return_last_out:
            last_out = self.base_model.forward_classifier(last_decoderfeats)
            if return_sim:
                return out, last_out, sim
            return out, last_out
        return out
    
    
    def infer_frame(self, frame, homo=None, last_frame=None, last_decoderfeats=None, last_feats=None):
        with torch.no_grad():
            feats = self.base_model.encoder(frame)
            decoderfeats = self.base_model.decoder_lastfeat(feats)
        
        if last_decoderfeats is not None:
            last_featmap, featmap = last_feats[0], feats[0]
            if homo is None:
                homo = self.registrator(last_frame, frame)
            
            with torch.no_grad():
                last_decoderfeats_warped = warp(homo, last_decoderfeats, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)
                last_featmap_warped = warp(homo, last_featmap, scale=None if self.low_scale_homo else 1/4, interp_mode=self.features_interp_mode)

                last_decoderfeats_warped, decoderfeats = self.pred_mixing(last_decoderfeats_warped, decoderfeats)
                sim = self.sim(last_featmap_warped, featmap, size=decoderfeats.shape[-2:])
                #decoderfeats = sim*last_decoderfeats_warped + (1-sim)*decoderfeats
                out = self.base_model.forward_classifier(decoderfeats)
                last_out_warped = self.base_model.forward_classifier(last_decoderfeats_warped)
                out = sim*last_out_warped + (1-sim)*out
        else:
            out = self.base_model.forward_classifier(decoderfeats)

        return out, feats, decoderfeats
    
    def infer_video(self, frames, homos, K, device):
        self.eval()
        preds = []
        last_frame = None
        last_decoderfeats = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            homo = homos[i]
            if homo is not None:
                homo = homo.to(device)
            out, feats, decoderfeats = self.infer_frame(frame, homo, last_frame, last_decoderfeats, last_feats)
            last_feats = feats
            last_decoderfeats = decoderfeats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class DFFWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.train_flow = model_cfg.get("train_flow", True)
        self.n_iters = model_cfg.get("flow_iters", 10)
        self.k_interval = model_cfg.get("k_interval", 4)
        self.flow_mode = model_cfg.get("flow_mode", "bilinear")


    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]
            out = self.base_model.decoder(last_feats_warped)

        if return_last_out:
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=True, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, last_frame=None, last_feats=None):
        if last_feats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]
                out = self.base_model.decoder(last_feats_warped)
            
            return out, last_feats_warped

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                out = self.base_model.decoder(feats)

            return out, feats
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i%self.k_interval == 0:
                out, feats = self.infer_frame(frame, last_frame, last_feats=None)
            else:
                out, feats = self.infer_frame(frame, last_frame, last_feats=last_feats)
            last_feats = feats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class DFF2Wrapper(DFFWrapper):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__(base_model, flow_model, n_classes, model_cfg)
    
    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            last_feats = self.base_model.encoder(last_frames)
            last_decoderfeats = self.base_model.decoder_lastfeat(last_feats)
            last_out = self.base_model.forward_classifier(last_decoderfeats)
            last_decoderfeats_warped = flowwarp(last_decoderfeats, interpolate_flow(flow, 1/4), self.flow_mode)
            out = self.base_model.forward_classifier(last_decoderfeats_warped)

        if return_last_out:
            return out, last_out
        return out
    
    def infer_frame(self, frame, last_frame=None, last_decoderfeats=None):
        if last_decoderfeats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                last_decoderfeats_warped = flowwarp(last_decoderfeats, interpolate_flow(flow, 1/4), self.flow_mode)
                out = self.base_model.forward_classifier(last_decoderfeats_warped)

            return out, last_decoderfeats_warped

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                decoderfeats = self.base_model.decoder_lastfeat(feats)
                out = self.base_model.forward_classifier(decoderfeats)

            return out, decoderfeats
    
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_decoderfeats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i%self.k_interval == 0:
                out, decoderfeats = self.infer_frame(frame, last_frame, last_decoderfeats=None)
            else:
                out, decoderfeats = self.infer_frame(frame, last_frame, last_decoderfeats=last_decoderfeats)
            last_decoderfeats = decoderfeats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds


class FlowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 2, 3, padding=1),
        )


    def forward(self, flow, frame, last_frame):
        x = torch.cat([flow, last_frame, frame, last_frame-frame], dim=1)
        x = self.conv_1(x)
        x = torch.cat([x, flow], dim=1)
        x = self.conv_2(x)
        return x


class NetWarpWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.train_flow = model_cfg.get("train_flow", True)
        self.n_iters = model_cfg.get("flow_iters", 8)
        self.flow_mode = model_cfg.get("flow_mode", "bilinear")
        self.flow_cnn = FlowCNN()
        self.past_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 96, 1, 1)),
            nn.Parameter(torch.zeros(1, 192, 1, 1)),
            nn.Parameter(torch.zeros(1, 384, 1, 1)),
            nn.Parameter(torch.zeros(1, 768, 1, 1))
        ])
        self.current_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 96, 1, 1)),
            nn.Parameter(torch.ones(1, 192, 1, 1)),
            nn.Parameter(torch.ones(1, 384, 1, 1)),
            nn.Parameter(torch.ones(1, 768, 1, 1))
        ])

    def forward(self, frames, last_frames, eval=False, return_last_out=False):
        if self.train_flow:
            self.flow_model.train()
            _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
        else:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frames, last_frames, iters=self.n_iters, test_mode=True)
                flow = flow.detach()

        with conditional_no_grad(eval):
            flow = self.flow_cnn(flow, frames, last_frames)

            last_feats = self.base_model.encoder(last_frames)
            last_out = self.base_model.decoder(last_feats)
            feats = self.base_model.encoder(frames)

            last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]

            combined_feats = []
            for k in range(len(feats)):
                combined_feats.append(self.current_weights[k] * feats[k] + self.past_weights[k] * last_feats_warped[k])

            out = self.base_model.decoder(combined_feats)

        if return_last_out:
            return out, last_out
        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=False, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            last_frames = adj_frames[0]
            out, last_out = self.forward(frames, last_frames, eval=True, return_last_out=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)
                last_out = nn.functional.interpolate(last_out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)
            if self.const_loss_lb > 0:
                const = self.const_loss_lb * self.const_loss(out, last_out, self.flow_model, frames, last_frames)
                loss += const

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, last_frame=None, last_feats=None):
        if last_feats is not None:
            with torch.no_grad():
                self.flow_model.eval()
                _, flow = self.flow_model(frame, last_frame, iters=self.n_iters, test_mode=True)
                flow = flow.detach()
            
            with torch.no_grad():
                flow = self.flow_cnn(flow, frame, last_frame)

                feats = self.base_model.encoder(frame)
                last_feats_warped = [flowwarp(f, interpolate_flow(flow, f.shape[-1]/flow.shape[-1]), self.flow_mode) for f in last_feats]

                combined_feats = []
                for k in range(len(feats)):
                    combined_feats.append(self.current_weights[k] * feats[k] + self.past_weights[k] * last_feats_warped[k])

                out = self.base_model.decoder(combined_feats)
            
            return out, combined_feats

        else:
            with torch.no_grad():
                feats = self.base_model.encoder(frame)
                out = self.base_model.decoder(feats)

            return out, feats
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        last_frame = None
        last_feats = None
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            out, feats = self.infer_frame(frame, last_frame, last_feats=last_feats)
            last_feats = feats
            last_frame = frame

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds
    

class TCBWrapper(nn.Module):
    def __init__(
            self, 
            base_model,
            flow_model, 
            n_classes, 
            model_cfg,
            ):
        super().__init__()
        self.base_model = base_model
        self.flow_model = flow_model
        self.n_classes = n_classes
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        # Losses
        self.const_loss_lb = model_cfg.get("const_loss_lb", 0)
        self.const_loss = TempConstLoss()
        self.tcb = Clip_PSP_Block(768, 768) if model_cfg.get("tcb_module", "ocr") == "psp" else Clip_OCR_Block(768, 768, self.n_classes, use_memory=False)
        self.tcb_conv = nn.Conv2d(768*2,768,1)

    def forward(self, frames, adj_frames, eval=False):
        with conditional_no_grad(eval):
            B = frames.size(0)
            all_frames = torch.cat([frames] + adj_frames, dim=0)
            all_feats = self.base_model.encoder(all_frames)
            feats = [all_f[:B] for all_f in all_feats]
            adj_feats = [all_f[B:].split(B, dim=0) for all_f in all_feats] # backprop on the adj frames

            featmap = feats[-1]
            adj_featmaps = list(adj_feats[-1])
            featmap_tcb = self.tcb(featmap, adj_featmaps)
            featmap = self.tcb_conv(torch.cat([featmap, featmap_tcb], dim=1))
            feats[-1] = featmap

            out = self.base_model.decoder(feats)

        return out
    
    def train_one_epoch(
            self,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
    ):
        train_loss = 0
        const_loss = 0
        const = None
        train_iter = tqdm(train_loader)
        self.train()
        for (frames, adj_frames, labels, homos) in train_iter:
            optimizer.zero_grad()
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)

            out = self.forward(frames, adj_frames, eval=False)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            train_iter.set_description(desc=f"train loss = {loss.item():.4f}, const = {const_value:.4f}")
        train_loss = train_loss/len(train_loader)
        const_loss = const_loss/len(train_loader)
        return train_loss, const_loss
    
    def evaluate_with_metrics(
            self,
            val_loader,
            criterion,
            device,
            n_classes,
            ignore_index=255
    ):
        val_loss = 0
        const_loss = 0
        const = None
        val_iter = tqdm(val_loader)
        mIoU1 = MetricMeter()
        mIoU2 = MetricMeter()
        #classes_mIoU = PerClassMetricMeter(n_classes)
        preds_for_iou = []
        labels_for_iou = []
        self.eval()
        for (frames, adj_frames, labels, homos) in val_iter:
            adj_frames = [adj_f.to(device) for adj_f in adj_frames]
            frames = frames.to(device)
            labels = labels.to(device)
            
            out = self.forward(frames, adj_frames, eval=True)

            if out.shape[-2:] != frames.shape[-2:]:
                out = nn.functional.interpolate(out, size=frames.shape[-2:], mode=self.upsampling, align_corners=False)

            loss = criterion(out, labels)

            preds = out.argmax(1)
            preds = preds.detach().cpu()
            if labels.dim() != preds.dim():
                labels = soft_to_hard_labels(labels, ignore_index)
            labels = labels.detach().cpu()
            preds_for_iou.append(preds)
            labels_for_iou.append(labels)

            val_loss += loss.item()
            const_value = const.item() if const is not None else 0
            const_loss += const_value
            val_iter.set_description(desc=f"val loss = {loss.item():.4f}, const = {const_value:.4f}")

        preds_for_iou = torch.cat(preds_for_iou, dim=0)
        labels_for_iou = torch.cat(labels_for_iou, dim=0)
        global_miou = meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        global_classes_iou = class_meanIoU(preds_for_iou, labels_for_iou, n_classes, ignore_index=ignore_index)
        val_loss = val_loss/len(val_loader)
        const_loss = const_loss/len(val_loader)

        results = {
            "val_loss": val_loss,
            "const_loss": const_loss,
            "mIoU1": mIoU1.avg,
            "mIoU2": mIoU2.avg,
            "global_miou": global_miou,
            "classes_mIoU": global_classes_iou
        }

        return results
    
    
    def infer_frame(self, frame, adj_frames=None):
        if adj_frames is not None:
            with torch.no_grad():
                B = frame.size(0)
                all_frames = torch.cat([frame] + adj_frames, dim=0)
                all_feats = self.base_model.encoder(all_frames)
                feats = [all_f[:B] for all_f in all_feats]
                adj_feats = [all_f[B:].detach().split(B, dim=0) for all_f in all_feats]

                featmap = feats[-1]
                adj_featmaps = list(adj_feats[-1])
                featmap_tcb = self.tcb(featmap, adj_featmaps)
                featmap = self.tcb_conv(torch.cat([featmap, featmap_tcb], dim=1))
                feats[-1] = featmap

                out = self.base_model.decoder(feats)
            
            return out

        else:
            with torch.no_grad():
                out = self.base_model(frame)

            return out
    
    def infer_video(self, frames, homos, device):
        self.eval()
        preds = []
        for (i, frame) in enumerate(frames):
            frame = frame.unsqueeze(0).to(device)
            if i>2:
                adj_frames = [frames[i-j].unsqueeze(0).to(device) for j in range(1,4)]
                out = self.infer_frame(frame, adj_frames=adj_frames)
            else:
                out = self.infer_frame(frame, adj_frames=None)

            if out.shape[-2:] != frame.shape[-2:]:
                out = nn.functional.interpolate(out, size=frame.shape[-2:], mode=self.upsampling, align_corners=False)
            pred = out.argmax(1).squeeze().detach().cpu()
            preds.append(pred)
        return preds