# data/dataset.py
import os
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
            # 兼容 ViTImageProcessor 和 torchvision transform
            processed = self.processor(images=image, return_tensors="pt")
            if isinstance(processed, dict):
                pixel_values = processed['pixel_values'].squeeze(0)
            else:
                pixel_values = processed
            return pixel_values, label
        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}, 错误: {e}")
            raise e