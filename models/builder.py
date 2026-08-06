#models/builder.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

class FeatureExtractorForClassification(nn.Module):
    """
    通用特征提取器封装：用于将无原生分类头的 Backbone (如 CLIP, DINOv2, Depth Anything, SAM2, SigLIP 等)
    转换为标准图像分类模型。
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 自动推断隐藏层特征维度
        hidden_size = None
        if hasattr(self.backbone.config, "projection_dim"):
            hidden_size = self.backbone.config.projection_dim
        elif hasattr(self.backbone.config, "hidden_size"):
            hidden_size = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "embed_dim"):
            hidden_size = self.backbone.config.embed_dim
        else:
            hidden_size = 768  # 兜底默认值

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        
        # 兼容不同的特征输出格式
        if hasattr(outputs, "image_embeds"):
            embeds = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeds = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            embeds = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, torch.Tensor):
            embeds = outputs.mean(dim=[2, 3]) if outputs.dim() == 4 else outputs
        else:
            embeds = outputs[0].mean(dim=1)

        return self.classifier(embeds)


def build_model_and_processor(model_name: str, num_classes: int, class_names: list):
    """
    统一的模型与预处理入口，自动判断模型类别并构建适配管道
    """
    model_name_lower = model_name.lower()

    # 1. HuggingFace 特征提取类模型 (CLIP, DINO, SigLIP, Depth Anything, SAM2, MobileCLIP 等)
    feature_models = ["clip", "dino", "siglip", "depth-anything", "sam2", "mobileclip", "tinyvim"]
    if any(k in model_name_lower for k in feature_models):
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            # 预处理 fallback
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            processor = lambda imgs, return_tensors=None: {'pixel_values': torch.stack([transform(img) for img in imgs]) if isinstance(imgs, list) else transform(imgs)}

        model = FeatureExtractorForClassification(model_name, num_classes)
        return model, processor

    # 2. 标准 HuggingFace 图像分类模型 (如 google/vit, EfficientFormer 等)
    elif "/" in model_name and not any(k in model_name_lower for k in ["timm/", "ultralytics"]):
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        id2label = {i: name for i, name in enumerate(class_names)} if class_names else {i: f"LABEL_{i}" for i in range(num_classes)}
        label2id = {name: i for i, name in enumerate(class_names)} if class_names else {f"LABEL_{i}": i for i in range(num_classes)}

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True
        )
        return model, processor

    # 3. timm 库注册模型 (ResNet, Swin, ConvNeXt, EfficientViT, RepViT, StarNet, MobileNetV4, EVA02, MambaVision, InternImage, FastViT 等)
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

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, trust_remote_code=True)
        return model, processor