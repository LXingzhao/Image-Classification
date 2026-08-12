# utils/dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

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


class SDNET2018Dataset(Dataset):
    """
    SDNET2018 专用数据集读取器
    sub_type: 'D' (Bridge Decks), 'P' (Pavements), 'W' (Walls), 或 'ALL'
    """
    def __init__(self, root_dir, processor, sub_type='D'):
        self.root_dir = root_dir
        self.processor = processor
        self.sub_type = sub_type
        self.samples = []

        # 统一类别映射：有裂缝 (Crack) -> 1, 无裂缝 (Uncracked) -> 0
        # 也可以修改为 {"Uncracked": 0, "Crack": 1}
        self.classes = ["Uncracked", "Crack"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 确定需要扫描的结构目录范围 ('D', 'P', 'W')
        if sub_type in ['D', 'P', 'W']:
            target_sub_types = [sub_type]
        elif sub_type.upper() == 'ALL':
            target_sub_types = ['D', 'P', 'W']
        else:
            raise ValueError(f"不支持的 sub_type: {sub_type}，可选为 'D', 'P', 'W', 'ALL'")

        # 遍历选定的子类型目录
        for st in target_sub_types:
            st_folder = os.path.join(root_dir, st)
            if not os.path.exists(st_folder):
                continue
            
            # 遍历 C* (Crack) 和 U* (Uncracked) 目录
            for folder_name in os.listdir(st_folder):
                folder_path = os.path.join(st_folder, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                
                # 依据文件夹开头字母判断类别: C* 为 Crack(1), U* 为 Uncracked(0)
                if folder_name.upper().startswith('C'):
                    label = self.class_to_idx["Crack"]
                elif folder_name.upper().startswith('U'):
                    label = self.class_to_idx["Uncracked"]
                else:
                    continue  # 跳过不符合命名规则的夹

                # 收集图片文件路径
                for fname in os.listdir(folder_path):
                    fpath = os.path.join(folder_path, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.samples.append((fpath, label))

        print(f"[SDNET2018 - {sub_type}] 加载完成，总样本量: {len(self.samples)} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 执行预处理 (兼容 HuggingFace Processor 与 torchvision transforms)
            processed = self.processor(images=image, return_tensors="pt") if callable(self.processor) else self.processor(image)
            
            # 兼容性解析逻辑
            if hasattr(processed, "pixel_values"):
                pixel_values = processed.pixel_values
            elif isinstance(processed, dict) and "pixel_values" in processed:
                pixel_values = processed["pixel_values"]
            else:
                pixel_values = processed

            # 去除多余的 batch 维度 (1, C, H, W) -> (C, H, W)
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
                pixel_values = pixel_values.squeeze(0)

            return pixel_values, label

        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}, 错误: {e}")
            raise e