# models/builder.py
import torch
import torchvision.transforms as transforms
import timm
from transformers import ViTForImageClassification, ViTImageProcessor

def build_model_and_processor(model_name: str, num_classes: int, class_names: list):
    """
    统一模型与预处理构建入口：支持 HuggingFace ViT 及 timm 通用 CNN (如 ResNet50)
    """
    # 1. 如果是 HuggingFace 的 ViT 模型
    if "google/vit" in model_name:
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        # --- 修改核心：判断 class_names 是否为空，避免传入空字典引发 Warning ---
        if class_names:
            id2label = {i: name for i, name in enumerate(class_names)}
            label2id = {name: i for i, name in enumerate(class_names)}
        else:
            id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
            label2id = {f"LABEL_{i}": i for i in range(num_classes)}

        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True, # 允许替换分类头
            id2label=id2label,            # 动态匹配标签字典
            label2id=label2id
        )
        return model, processor

    # 2. 如果是通用 CNN 结构（如 ResNet50）
    else:
        # 定义标准 ResNet 图像预处理流水线 (224x224 尺寸 + ImageNet 归一化)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 封装一个兼容 processor 接口的简单处理函数
        def processor(images, return_tensors=None):
            if isinstance(images, list):
                pixel_values = torch.stack([transform(img) for img in images])
            else:
                pixel_values = transform(images)
            return {'pixel_values': pixel_values}

        # 使用 timm 加载预训练 ResNet50
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        return model, processor