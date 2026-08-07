# models/builder.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from timm.data import create_transform, resolve_model_data_config
from transformers import (
    PreTrainedModel,
    AutoImageProcessor, 
    AutoModel, 
    AutoModelForImageClassification, 
    AutoModelForDepthEstimation,
    AutoConfig
)

# 修复新版 transformers 与 OpenGVLab 远程代码的兼容性 bug
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: getattr(self, "_tied_weights_keys", None) or {}
    )


class FeatureExtractorForClassification(nn.Module):
    """
    通用特征提取器封装：用于将无原生分类头的 Backbone (如 CLIP, DINOv2, Depth Anything, SAM2, SigLIP, InternImage 等)
    转换为标准图像分类模型。
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.is_custom_classifier = False  # 标记是否需要自建分类头
        
        # 1. 优先尝试 AutoModelForImageClassification 加载 (兼容 InternImage / ViT / ResNet 等)
        try:
            full_model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            )
            self.backbone = full_model
            self.is_custom_classifier = False
            
        except Exception:
            # 2. 如果失败，退回使用标准 AutoModel (适用于 CLIP, DINOv2, SAM2 等双塔或无分类头模型)
            try:
                full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                if hasattr(full_model, "vision_model"):
                    self.backbone = full_model.vision_model
                else:
                    self.backbone = full_model
                self.is_custom_classifier = True

            except ValueError:
                # 3. Depth Anything 等深度估计模型，提取内部特征 backbone
                try:
                    depth_model = AutoModelForDepthEstimation.from_pretrained(model_name, trust_remote_code=True)
                    if hasattr(depth_model, "backbone"):
                        self.backbone = depth_model.backbone
                    else:
                        self.backbone = depth_model
                    self.is_custom_classifier = True

                except Exception:
                    # 4. 尝试通过 timm 库进行兜底加载
                    try:
                        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
                        self.is_custom_classifier = False
                    except Exception:
                        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                        self.backbone = AutoModel.from_config(config, trust_remote_code=True)
                        self.is_custom_classifier = True

        # 如果使用的不是 AutoModelForImageClassification 或 timm，需要手动构建Linear分类头
        if self.is_custom_classifier:
            embed_dim = self._get_embed_dim()
            self.classifier = nn.Linear(embed_dim, num_classes)

    def _get_embed_dim(self):
        """自动推断 backbone 的特征输出维度"""
        if hasattr(self.backbone, "config"):
            cfg = self.backbone.config
            for attr in ["hidden_size", "d_model", "embed_dim", "projection_dim", "num_features"]:
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        # 如果从 config 拿不到，给出一个常用默认值
        return getattr(self.backbone, "num_features", 768)

    def forward(self, x):
        if not self.is_custom_classifier:
            # 由 AutoModelForImageClassification 或 timm 直接处理 forward 并输出 logits
            outputs = self.backbone(x)
            return outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            # 提取特征后送入自建的 classifier
            outputs = self.backbone(x)
            if hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state
                if feat.dim() == 4:
                    feat = feat.mean(dim=[-2, -1])  # GAP 全局池化
                elif feat.dim() == 3:
                    feat = feat[:, 0, :]           # 取 [CLS] token
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                feat = outputs
                if feat.dim() == 4:
                    feat = feat.mean(dim=[-2, -1])
            return self.classifier(feat)


def build_model_and_processor(model_name: str, num_classes: int, class_names: list):
    """
    统一的模型与预处理入口，自动解析模型自身所需的分辨率 (Img Size)
    """
    model_name_lower = model_name.lower()

    # 标准通用图像 Transform 兜底处理
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def fallback_processor(images=None, imgs=None, return_tensors=None, **kwargs):
        target_imgs = images if images is not None else imgs
        if target_imgs is None:
            raise ValueError("No images passed to processor")
        if isinstance(target_imgs, list):
            pixel_values = torch.stack([default_transform(img) for img in target_imgs])
        else:
            pixel_values = default_transform(target_imgs)
        return {'pixel_values': pixel_values}

    # 1. Ultralytics YOLO 系列分类模型
    if "yolo" in model_name_lower or "ultralytics" in model_name_lower:
        from ultralytics import YOLO
        
        yolo_obj = YOLO(model_name)
        base_model = yolo_obj.model
        
        if hasattr(base_model, 'nc'):
            base_model.nc = num_classes
            
        last_layer = base_model.model[-1]
        if hasattr(last_layer, 'linear'):
            in_features = last_layer.linear.in_features
            last_layer.linear = nn.Linear(in_features, num_classes)
        elif hasattr(last_layer, 'cv2'):
            in_features = last_layer.cv2.conv.in_channels
            last_layer.linear = nn.Linear(in_features, num_classes)

        class YOLOClassificationWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                outputs = self.model(pixel_values)
                if isinstance(outputs, (tuple, list)):
                    return outputs[0]
                return outputs

            def load_state_dict(self, state_dict, strict=True):
                first_key = next(iter(state_dict.keys()))
                if not first_key.startswith("model."):
                    state_dict = {f"model.{k}": v for k, v in state_dict.items()}
                return super().load_state_dict(state_dict, strict=strict)

        model = YOLOClassificationWrapper(base_model)
        return model, fallback_processor

# 2. HuggingFace 特征提取类模型
    feature_models = ["clip", "dino", "siglip", "depth-anything", "sam2", "mobileclip", "tinyvim", "internimage"]
    if any(k in model_name_lower for k in feature_models):
        try:
            # 针对 sam2 和 depth-anything 等高分辨率模型，强制覆盖输入分辨率为 224x224
            if "sam2" in model_name_lower or "depth-anything" in model_name_lower:
                processor = AutoImageProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    size={"height": 224, "width": 224}
                )
            else:
                processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            processor = fallback_processor

        model = FeatureExtractorForClassification(model_name, num_classes)
        return model, processor

    # 3. 标准 HuggingFace 图像分类模型
    elif "/" in model_name and not any(k in model_name_lower for k in ["timm/", "ultralytics"]) and not model_name.startswith("hf_hub:"):
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            processor = fallback_processor

        id2label = {i: name for i, name in enumerate(class_names)} if class_names else {i: f"LABEL_{i}" for i in range(num_classes)}
        label2id = {name: i for i, name in enumerate(class_names)} if class_names else {f"LABEL_{i}": i for i in range(num_classes)}

        hf_model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    id2label=id2label,
                    label2id=label2id,
                    trust_remote_code=True
                )

        class HFClassificationWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values, **kwargs):
                outputs = self.model(pixel_values=pixel_values, **kwargs)
                if hasattr(outputs, "logits"):
                    return outputs.logits
                elif isinstance(outputs, (tuple, list)):
                    return outputs[0]
                return outputs

        model = HFClassificationWrapper(hf_model)
        return model, processor

    # 4. timm 库注册模型
    else:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        except RuntimeError as e:
            err_str = str(e)
            if "Invalid pretrained tag" in err_str:
                arch_name = model_name.split('.')[0]
                available_models = timm.list_models(f"*{arch_name}*", pretrained=True)
                if available_models:
                    print(f"⚠️ [警告] Tag 无效: {model_name}，自动修正并加载可用权重: {available_models[0]}")
                    model = timm.create_model(available_models[0], pretrained=True, num_classes=num_classes)
                else:
                    raise e
            elif "Unknown model" in err_str and not model_name.startswith("hf_hub:"):
                print(f"⚠️ [警告] timm 未查找到模型 {model_name}，尝试通过 HuggingFace 加载...")
                try:
                    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                except Exception:
                    processor = fallback_processor

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
            else:
                raise e
        
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)

        def processor(images, return_tensors=None):
            if isinstance(images, list):
                pixel_values = torch.stack([transform(img) for img in images])
            else:
                pixel_values = transform(images)
            return {'pixel_values': pixel_values}

        return model, processor