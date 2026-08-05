# utils/dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CambridgeBridgeDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.samples = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                fpath = os.path.join(cls_folder, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, self.class_to_idx[cls_name]))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 执行预处理 (兼容 HuggingFace Processor 与 torchvision transforms)
            processed = self.processor(images=image, return_tensors="pt") if callable(self.processor) else self.processor(image)
            
            # --- 兼容性解析逻辑 ---
            # 情况 1: HuggingFace 返回 BatchFeature 或 dict (如 ViT, Swin)
            if hasattr(processed, "pixel_values"):
                pixel_values = processed.pixel_values
            elif isinstance(processed, dict) and "pixel_values" in processed:
                pixel_values = processed["pixel_values"]
            # 情况 2: torchvision/timm 返回 torch.Tensor (如 ResNet50, ConvNeXt)
            else:
                pixel_values = processed

            # 确保 Tensor 形状为 (C, H, W)，去除多余的 (1, C, H, W) 维度
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
                pixel_values = pixel_values.squeeze(0)

            return pixel_values, label

        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}, 错误: {e}")
            raise e