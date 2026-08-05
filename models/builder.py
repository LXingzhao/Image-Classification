# models/builder.py
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

class CLIPForImageClassification(nn.Module):
    """封装 CLIP Vision Tower 用于图像分类"""
    def __init__(self, model_name, num_classes):
        super().__init__()
        # 加载预训练的 CLIP 视觉投影模型
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(model_name)
        # 获取图像特征向量维度 (CLIP ViT-L/14 投影后维度为 768)
        hidden_size = self.clip_vision.config.projection_dim
        # 自定义线性分类头
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # 提取图像特征
        outputs = self.clip_vision(pixel_values=pixel_values)
        image_embeds = outputs.image_embeds  # [batch_size, hidden_size]
        logits = self.classifier(image_embeds)
        return logits


def build_model_and_processor(model_name: str, num_classes: int, class_names: list):
    """
    统一模型与预处理构建入口：支持 CLIP ViT、HuggingFace ViT 及 timm CNN
    """
    # 1. 如果是 CLIP 模型
    if "clip" in model_name.lower():
        processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPForImageClassification(model_name, num_classes)
        return model, processor

    # 2. 如果是标准 HuggingFace ViT 模型
    elif "google/vit" in model_name.lower():
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        if class_names:
            id2label = {i: name for i, name in enumerate(class_names)}
            label2id = {name: i for i, name in enumerate(class_names)}
        else:
            id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
            label2id = {f"LABEL_{i}": i for i in range(num_classes)}

        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id
        )
        return model, processor

    # 3. 通用 CNN (如 ResNet)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        def processor(images, return_tensors=None):
            if isinstance(images, list):
                pixel_values = torch.stack([transform(img) for img in images])
            else:
                pixel_values = transform(images)
            return {'pixel_values': pixel_values}

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model, processor