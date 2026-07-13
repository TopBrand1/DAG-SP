import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.video.mixing_layers import (
    STT3DMixingLayer,
    STT2DMixingLayer,
    Conv2DMixingLayer,
)

def _pack_attention(sim, return_attention=False, method="unknown", **kwargs):
    """
    Backward-compatible attention output.

    return_attention=False:
        return sim only, unchanged for training/inference.

    return_attention=True:
        return (sim, attention_dict)
    """
    if not return_attention:
        return sim

    attn = {
        "method": method,
        "sim": sim
    }
    for k, v in kwargs.items():
        if v is not None:
            attn[k] = v
    return sim, attn

# Constant interpolation. If alpha=1, no interpolation, only current prediction.
class ConstantSimBlock(nn.Module):
    def __init__(self, alpha):
        super(ConstantSimBlock, self).__init__()
        self.alpha = alpha

    def forward(self, last_featmap, featmap, size):
        sim = torch.ones((featmap.size(0), 1, size[0], size[1]), dtype=torch.float32, device=featmap.device) * self.alpha
        return sim

# Constant interpolation with learnable value. Starts at 0.5 for average of both predictions
class AverageSimBlock(nn.Module):
    def __init__(self, alpha):
        super(AverageSimBlock, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, last_featmap, featmap, size):
        sim = torch.ones((featmap.size(0), 1, size[0], size[1]), dtype=torch.float32, device=featmap.device) * self.alpha
        return sim

# Cosine similarity on features
class ConvSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(embed_dim//2, 1, 1)

    def forward(self, last_featmap, featmap, size, return_attention=False):
        x = torch.cat([last_featmap, featmap], dim=1)
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        sim = self.conv3(feat2).sigmoid()

        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)

        # For visualization only: averaged convolutional activation.
        conv_activation = feat2.mean(dim=1, keepdim=True)

        return _pack_attention(
            sim,
            return_attention=return_attention,
            method="conv_similarity",
            conv_activation=conv_activation
        )

class TFCCosSimBlock(nn.Module):
    def __init__(self, eps=1e-6, power=1.0, max_value=1.0):
        super(TFCCosSimBlock, self).__init__()
        self.eps = eps
        self.power = power
        self.max_value = max_value

    def forward(self, last_featmap, featmap, size, return_attention=False):
        if last_featmap.shape[-2:] != featmap.shape[-2:]:
            last_featmap = F.interpolate(
                last_featmap,
                size=featmap.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        last_norm = F.normalize(last_featmap, p=2, dim=1, eps=self.eps)
        feat_norm = F.normalize(featmap, p=2, dim=1, eps=self.eps)

        sim = torch.sum(last_norm * feat_norm, dim=1, keepdim=True)
        sim = torch.clamp(sim, min=0.0, max=1.0)

        if self.power != 1.0:
            sim = sim ** self.power

        if self.max_value < 1.0:
            sim = torch.clamp(sim, min=0.0, max=self.max_value)

        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)

        if return_attention:
            attention_info = {
                "method": "tfc_cosine_similarity",
                "sim": sim
            }
            return sim, attention_info

        return sim
    
# Convolutions on last features
class ConvSimBlockBase(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlockBase, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = last_featmap
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

# Convolutions on concatenated features 卷积块计算相似度图: 这个模块通过卷积层计算两个特征图之间的相似度。它接收两个特征图（last_featmap和featmap），将它们连接起来，然后通过几个卷积层和一个sigmoid激活函数，输出一个0到1之间的相似度图。
class ConvSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.cat([last_featmap, featmap], dim=1)
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# 简单相关性计算机制
class ImprovedModulatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, attention_type="both"):
        super(ImprovedModulatedCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.attention_type = attention_type

        # 1. Feature fusion with dilated convolution
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(embed_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Spatial / local attention
        self.local_attention = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//4, 3, padding=1, groups=max(1, embed_dim//4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//4, embed_dim, 1),
            nn.Sigmoid()
        )

        # 3. Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, max(1, embed_dim//16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, embed_dim//16), embed_dim, 1),
            nn.Sigmoid()
        )

        # 4. Residual branch
        self.residual_conv = nn.Conv2d(embed_dim, embed_dim, 1)

        # 5. Similarity output
        self.output_layers = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//4, 3, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//4, 1, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, last_featmap, featmap, size, return_attention=False):
        # 1. Concatenate cross-frame features
        x = torch.cat([last_featmap, featmap], dim=1)

        # 2. Multi-scale contextual fusion
        fused = self.feature_fusion(x)

        # 3. Spatial/local attention and channel attention
        local_attn = self.local_attention(fused)          # [B, C, H, W]
        channel_attn = self.channel_attention(fused)      # [B, C, 1, 1]

        # 4. Dual-attention modulation
        attended = fused * local_attn * channel_attn

        # 5. Residual-enhanced feature
        residual = self.residual_conv(fused)
        x_attended = attended + residual

        # 6. Similarity prediction
        sim = self.output_layers(x_attended)

        # Keep the original computation behavior.
        sim = torch.sigmoid(sim * self.temperature)

        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)

        # Visualization maps
        # spatial_attn is the channel-averaged local attention.
        spatial_attn = local_attn.mean(dim=1, keepdim=True)

        # channel_activation projects channel modulation back to 2D.
        channel_activation = (fused * channel_attn).mean(dim=1, keepdim=True)

        dual_attention_activation = attended.mean(dim=1, keepdim=True)

        return _pack_attention(
            sim,
            return_attention=return_attention,
            method="dual_attention",
            spatial_attn=spatial_attn,
            channel_attn=channel_attn,
            channel_activation=channel_activation,
            dual_attention_activation=dual_attention_activation
        )

   
    
class MultiScaleLocalCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv3x3 = nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1)
        self.conv5x5 = nn.Conv2d(embed_dim*2, embed_dim, 5, padding=2)
        self.conv7x7 = nn.Conv2d(embed_dim*2, embed_dim, 7, padding=3)

        self.scale_selector = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 3, 1),
            nn.Softmax(dim=1)
        )

        self.local_corr_conv = nn.Conv2d(embed_dim, embed_dim, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_local_correlation(self, last_featmap, featmap):
        correlation = self.local_corr_conv(featmap)
        correlation = correlation * last_featmap
        return correlation

    def forward(self, last_featmap, featmap, size, return_attention=False):
        B, C, H, W = featmap.shape

        if C != self.embed_dim:
            if hasattr(self, 'adapt_conv'):
                featmap = self.adapt_conv(featmap)
                last_featmap = self.adapt_conv(last_featmap)
            else:
                self.adapt_conv = nn.Conv2d(C, self.embed_dim, 1).to(featmap.device)
                nn.init.kaiming_normal_(self.adapt_conv.weight, mode='fan_out', nonlinearity='relu')
                featmap = self.adapt_conv(featmap)
                last_featmap = self.adapt_conv(last_featmap)

        combined = torch.cat([last_featmap, featmap], dim=1)

        f3 = self.conv3x3(combined)
        f5 = self.conv5x5(combined)
        f7 = self.conv7x7(combined)

        scale_weights = self.scale_selector(combined)
        f_multi = scale_weights[:, 0:1] * f3 + scale_weights[:, 1:2] * f5 + scale_weights[:, 2:3] * f7

        local_corr = self.compute_local_correlation(last_featmap, featmap)

        final_feat = torch.cat([f_multi, local_corr], dim=1)
        sim = self.fusion(final_feat)

        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)

        scale_activation = f_multi.mean(dim=1, keepdim=True)
        local_corr_activation = local_corr.mean(dim=1, keepdim=True)

        return _pack_attention(
            sim,
            return_attention=return_attention,
            method="multi_scale_local_cross_attention",
            scale_weights=scale_weights,
            scale_activation=scale_activation,
            local_corr_activation=local_corr_activation
        )
    
class SimplifiedModulatedAttentionSimBlock(nn.Module):
    """Simplified modulated attention similarity block."""
    def __init__(self, embed_dim, reduction_ratio=4):
        super(SimplifiedModulatedAttentionSimBlock, self).__init__()
        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 2, embed_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU()
        )

        self.sim_output = nn.Sequential(
            nn.Conv2d(embed_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        self.residual_conv = nn.Conv2d(embed_dim * 2, embed_dim // 2, 1)

    def forward(self, last_featmap, featmap, size, return_attention=False):
        channel_weights = self.channel_attention(
            torch.cat([last_featmap, featmap], dim=1)
        )

        last_featmap_weighted = last_featmap * channel_weights
        featmap_weighted = featmap * channel_weights

        feat_diff = torch.abs(last_featmap_weighted - featmap_weighted)

        avg_pool = torch.mean(feat_diff, dim=1, keepdim=True)
        max_pool, _ = torch.max(feat_diff, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))

        attended_diff = feat_diff * spatial_weights

        fused_features = self.feature_fusion(
            torch.cat([last_featmap_weighted, featmap_weighted], dim=1)
        )

        enhanced_features = fused_features + attended_diff.mean(dim=1, keepdim=True)

        residual = self.residual_conv(
            torch.cat([last_featmap, featmap], dim=1)
        )
        final_features = enhanced_features + residual

        sim = self.sim_output(final_features)

        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)

        channel_activation = (featmap * channel_weights).mean(dim=1, keepdim=True)

        return _pack_attention(
            sim,
            return_attention=return_attention,
            method="simplified_modulated_attention",
            spatial_attn=spatial_weights,
            channel_attn=channel_weights,
            channel_activation=channel_activation
        )
    
    
class ConstantSimBlock(nn.Module):
    def __init__(self, constant_value):
        super(ConstantSimBlock, self).__init__()
        self.constant_value = constant_value

    def forward(self, last_featmap, featmap, size):
        batch_size, _, h, w = last_featmap.shape
        sim = torch.ones(batch_size, 1, h, w, device=last_featmap.device) * self.constant_value
        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

class AverageSimBlock(nn.Module):
    def __init__(self, constant_value):
        super(AverageSimBlock, self).__init__()
        self.constant_value = constant_value

    def forward(self, last_featmap, featmap, size):
        batch_size, _, h, w = last_featmap.shape
        sim = torch.ones(batch_size, 1, h, w, device=last_featmap.device) * self.constant_value
        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

class CossimSimBlock(nn.Module):
    def __init__(self):
        super(CossimSimBlock, self).__init__()

    def forward(self, last_featmap, featmap, size):
        # 计算余弦相似度
        last_feat_flat = last_featmap.flatten(2)  # [B, C, H*W]
        feat_flat = featmap.flatten(2)            # [B, C, H*W]
        
        # 归一化
        last_feat_norm = F.normalize(last_feat_flat, p=2, dim=1)
        feat_norm = F.normalize(feat_flat, p=2, dim=1)
        
        # 计算相似度
        sim_matrix = torch.bmm(last_feat_norm.transpose(1, 2), feat_norm)  # [B, H*W, H*W]
        sim = sim_matrix.max(dim=2)[0].view(last_featmap.shape[0], 1, last_featmap.shape[2], last_featmap.shape[3])
        
        if sim.shape[-2:] != size:
            sim = F.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# Mixing of features with cnvolution then convolution on last features
class ConvSimBlockv2(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlockv2, self).__init__()
        self.convmix = Conv2DMixingLayer(embed_dim*2, hidden_dim=8*embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )
        self.embed_dim = embed_dim

    def forward(self, last_featmap, featmap, size):
        x = torch.cat([last_featmap, featmap], dim=1)
        x = self.convmix(x)
        x = x[:,:self.embed_dim]
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    

# 3D-WMSA mixing then convolution on last features
class STT3DConvSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DConvSimBlock, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        x = self.att(x)
        x = self.conv(rearrange(x[:,0], 'b h w c -> b c h w'))
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# 3D-WMSA mixing then convolution on concatenated features
class STT3DConvSimBlock2(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DConvSimBlock2, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        x = self.att(x)
        x = self.conv(torch.cat([rearrange(x[:,0], 'b h w c -> b c h w'), rearrange(x[:,1], 'b h w c -> b c h w')], dim=1))
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# 3D-WMSA mixing then cosine similarity
class STT3DCosSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DCosSimBlock, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.cossim = nn.CosineSimilarity(dim=1)

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        sim = self.cossim(rearrange(x[:,0], 'b h w c -> b c h w'), rearrange(x[:,1], 'b h w c -> b c h w'))
        sim = sim.unsqueeze(1)
        sim = torch.clamp(sim, min=0, max=1)**4
        sim = torch.clamp(sim, min=0, max=0.99)
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim