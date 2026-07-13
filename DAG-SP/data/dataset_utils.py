import os
from data.datasets import UAVID, RURALSCAPES

import cv2
import data.utils.images_transforms as image_transforms
import albumentations as aug
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

def to_float32(image, **kwargs):
    """Convert image to float32 with proper memory management"""
    result = image.astype(np.float32)
    return np.ascontiguousarray(result)

def ensure_contiguous(image, **kwargs):
    """Ensure memory contiguous (module-level function)"""
    return np.ascontiguousarray(image)

def parse_datasets(name, path=None, split="train"):
    name = name.lower()
    DATASET = None
    if name == "uavid":
        DATASET = UAVID
    elif name == "ruralscapes":
        DATASET = RURALSCAPES
    assert DATASET is not None, f"Dataset {name} not implemented"

    if path is None:
        path = DATASET.path
        
    data_folder = os.path.join(path, "data")

    def get_split_indices(split):
        with open(os.path.join(path, split)) as f:
            indices = f.readlines(-1)

        video_indices = []
        for idx in indices:
            v_idx = idx[:-1] # remove \n
            if v_idx not in video_indices:
                video_indices.append(v_idx)

        return video_indices

    if split == "test":
        video_train_idx = None
        video_val_idx = get_split_indices("test.txt")
    else:
        video_train_idx = get_split_indices("train.txt")
        video_val_idx = get_split_indices("val.txt")

    return DATASET, data_folder, video_train_idx, video_val_idx


def get_transforms(data_type, crop_size, DATASET, data_augmentation=True, soft_labels=False, square_crop=False):
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if data_type == "video":
        frame_transforms_val = image_transforms.VideoCompose([
                image_transforms.VideoCenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.VideoIdentity(),
                image_transforms.VideoNormalize(mean=normalize[0], std=normalize[1]),
                image_transforms.VideoToTensor(),
            ])

        mask_transforms = image_transforms.Compose([
                image_transforms.CenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.Identity(),
                image_transforms.LabelToTensor(n_classes=DATASET.n_classes, convert_labels=DATASET.convert_labels) if not soft_labels else image_transforms.ToTensor()
            ])
            
        if data_augmentation:
            augmentations = image_transforms.VideoCompose_wLabel([
                    image_transforms.VideoRandomHorizontalFlip(0.5),
                    image_transforms.VideoRandomScaleCrop(0.75, crop_size, scales=[1.1,1.15,1.2,1.25,1.3,1.35,1.4], interpolation="bilinear"),
                    image_transforms.VideoRandomCrop(1, crop_size[0]) if square_crop else image_transforms.VideoIdentity_wLabel()
                ])

            frame_transforms = image_transforms.VideoCompose([
                    image_transforms.VideoColorJitter(0.3),
                    image_transforms.VideoCenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.VideoIdentity(),
                    image_transforms.VideoNormalize(mean=normalize[0], std=normalize[1]),
                    image_transforms.VideoToTensor(),
                ])
        else:
            augmentations = None
            frame_transforms = frame_transforms_val

    # elif data_type == "image":
    #     transform_aug = aug.Compose(
    #         [
    #             aug.HorizontalFlip(p=0.5),
    #             aug.OneOf(
    #                 [
    #                     aug.Perspective(p=1, scale=(0.01, 0.05)),
    #                     aug.ShiftScaleRotate(
    #                         scale_limit=0.1,
    #                         rotate_limit=15,
    #                         shift_limit=0.0625,
    #                         interpolation=cv2.INTER_LINEAR,
    #                         border_mode=0,
    #                         # value=0,
    #                         # mask_value=0,
    #                         p=1.,
    #                     ),
    #                 ],
    #                 p=0.2
    #             ),
    #             aug.OneOf(
    #                 [
    #                     aug.RandomBrightnessContrast(p=0.5),
    #                     aug.HueSaturationValue(p=0.5),
    #                     aug.RandomGamma(p=0.5),
    #                     aug.CLAHE(p=0.5),
    #                 ],
    #                 p=0.8,
    #             ),
    #             aug.OneOf(
    #                 [
    #                     aug.ISONoise(p=0.5),
    #                     aug.GaussNoise(p=0.5),
    #                     aug.ImageCompression(p=0.5),
    #                     aug.Sharpen(p=0.5),
    #                 ],
    #                 p=0.6,
    #             ),
    #             aug.RandomCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
    #         ],)

    #     augmentations = transform_aug if data_augmentation else None

    #     normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     frame_transforms = aug.Compose([
    #         aug.CenterCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
    #         aug.Normalize(mean=normalize[0], std=normalize[1], max_pixel_value=255.),
    #         ToTensorV2(),
    #     ])
    #     frame_transforms_val = frame_transforms

    #     mask_transforms = image_transforms.Compose([
    #         image_transforms.CenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.Identity(),
    #         image_transforms.LabelToTensor(n_classes=DATASET.n_classes, convert_labels=DATASET.convert_labels) if not soft_labels else image_transforms.ToTensor()
    #     ])

    # return augmentations, frame_transforms, frame_transforms_val, mask_transforms
    elif data_type == "image":
        # === CRITICAL FIX: Albumentations 2.0.8 专用修复 ===
        # 1. 完全禁用 OpenCV 后端
        aug.augmentations.geometric.rotate._cv2 = None
        aug.augmentations.geometric.resize._cv2 = None
        aug.augmentations.geometric.functional._cv2 = None
        
        # 2. 使用 Albumentations 2.0.8 兼容的 Affine 参数
        transform_aug = aug.Compose(
            [
                aug.HorizontalFlip(p=0.5),
                aug.OneOf(
                    [
                        # 使用 Albumentations 2.0.8 推荐的参数
                        aug.Affine(
                            scale=(0.95, 1.05),
                            translate_percent=(-0.05, 0.05),
                            rotate=(-15, 15),
                            # 关键：使用 border_mode=0 (cv2.BORDER_REFLECT_101)
                            border_mode=0,
                            # 关键：显式指定 value=0
                            fill=0,
                            fill_mask=0,
                            p=1.0
                        ),
                        aug.Affine(
                            scale=(0.9, 1.1),
                            translate_percent=(-0.0625, 0.0625),
                            rotate=(-15, 15),
                            border_mode=0,
                            fill=0,
                            fill_mask=0,
                            p=1.0
                        ),
                    ],
                    p=0.2
                ),
                aug.OneOf(
                    [
                        aug.RandomBrightnessContrast(p=0.5),
                        aug.HueSaturationValue(p=0.5),
                        aug.RandomGamma(p=0.5),
                        aug.CLAHE(p=0.5),
                    ],
                    p=0.8,
                ),
                aug.OneOf(
                    [
                        aug.ISONoise(p=0.5),
                        aug.GaussNoise(p=0.5),
                        aug.ImageCompression(p=0.5),
                        aug.Sharpen(p=0.5),
                    ],
                    p=0.6,
                ),
                aug.RandomCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
            ],
            p=1.0,
            is_check_shapes=False
        )

        # 应用内存连续化
        transform_aug = aug.Compose([
            *transform_aug.transforms,
            aug.Lambda(image=ensure_contiguous, mask=ensure_contiguous)
        ], is_check_shapes=False)
        
        augmentations = transform_aug if data_augmentation else None

        # 修复 Normalize
        frame_transforms = aug.Compose([
            aug.CenterCrop(crop_size[0], crop_size[0]) if square_crop else aug.NoOp(),
            aug.Lambda(image=to_float32),
            aug.Normalize(
                mean=normalize[0], 
                std=normalize[1],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ], is_check_shapes=False)
        frame_transforms_val = frame_transforms

        mask_transforms = image_transforms.Compose([
            image_transforms.CenterCrop((crop_size[0], crop_size[0])) if square_crop else image_transforms.Identity(),
            image_transforms.LabelToTensor(n_classes=DATASET.n_classes, convert_labels=DATASET.convert_labels) if not soft_labels else image_transforms.ToTensor()
        ])

    return augmentations, frame_transforms, frame_transforms_val, mask_transforms