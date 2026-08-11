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

# 彻底兼容新版 transformers 与 OpenGVLab/ViT 远程/原生代码的 tied_weights 属性读写
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_tied_keys(self):
        return getattr(self, "_tied_weights_keys", None) or {}
    def _set_tied_keys(self, value):
        self._tied_weights_keys = value

    PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)

class FeatureExtractorForClassification(nn.Module):
    """
    通用特征提取器封装：用于将无原生分类头的 Backbone (如 CLIP, DINOv2, Depth Anything, SAM2, SigLIP, InternImage 等)
    转换为标准图像分类模型。
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.is_custom_classifier = False  # 标记是否需要自建分类头
        model_name_lower = model_name.lower()
        
        # 1. 针对 SAM2 系列进行精准匹配（解决 SAM2 加载报错与特征提取问题）
        if "sam2" in model_name_lower:
            try:
                # 优先尝试从 transformers 导入专业的 Sam2VisionModel
                from transformers import Sam2VisionModel
                self.backbone = Sam2VisionModel.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                # 兜底：如果 transformers 版本较低，使用 AutoModel 提取其中的 image_encoder
                full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                if hasattr(full_model, "image_encoder"):
                    self.backbone = full_model.image_encoder
                elif hasattr(full_model, "vision_model"):
                    self.backbone = full_model.vision_model
                else:
                    self.backbone = full_model
            self.is_custom_classifier = True

        else:
            # 2. 其他无分类头模型：首先判断是否已知是没有原生 Classification Head 的模型
            # 跳过 AutoModelForImageClassification 以避免产生控制台冗余报警日志
            known_non_classification_models = ["clip", "dino", "siglip", "depth-anything", "mobileclip"]
            should_skip_for_class = any(k in model_name_lower for k in known_non_classification_models)

            loaded = False
            if not should_skip_for_class:
                try:
                    full_model = AutoModelForImageClassification.from_pretrained(
                        model_name,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True
                    )
                    self.backbone = full_model
                    self.is_custom_classifier = False
                    loaded = True
                except Exception:
                    loaded = False

            if not loaded:
                # 3. 退回使用标准 AutoModel (适用于 CLIP, DINOv2 等双塔模型)
                try:
                    full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    if hasattr(full_model, "vision_model"):
                        self.backbone = full_model.vision_model
                    else:
                        self.backbone = full_model
                    self.is_custom_classifier = True

                except ValueError:
                    # 4. Depth Anything 等深度估计模型，提取内部特征 backbone
                    try:
                        depth_model = AutoModelForDepthEstimation.from_pretrained(model_name, trust_remote_code=True)
                        if hasattr(depth_model, "backbone"):
                            self.backbone = depth_model.backbone
                        else:
                            self.backbone = depth_model
                        self.is_custom_classifier = True

                    except Exception:
                        # 5. 尝试通过 timm 库进行兜底加载
                        try:
                            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
                            self.is_custom_classifier = False
                        except Exception:
                            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
                            self.is_custom_classifier = True

        # 如果使用的不是 AutoModelForImageClassification 或 timm，手动构建 Linear 分类头
        if self.is_custom_classifier:
            embed_dim = self._get_embed_dim()
            self.classifier = nn.Linear(embed_dim, num_classes)

    def _get_embed_dim(self):
            """自动推断 backbone 的特征输出维度"""
            if hasattr(self.backbone, "config"):
                cfg = self.backbone.config
                # 1. 优先识别 SAM2 的特征通道
                if hasattr(cfg, "output_channels") and isinstance(cfg.output_channels, (list, tuple)):
                    return cfg.output_channels[-1]
                if hasattr(cfg, "vision_config"):
                    vc = cfg.vision_config
                    if hasattr(vc, "output_channels") and isinstance(vc.output_channels, (list, tuple)):
                        return vc.output_channels[-1]
                    if hasattr(vc, "hidden_size"):
                        return vc.hidden_size
                if hasattr(cfg, "hidden_dim"):
                    return cfg.hidden_dim
                # 2. 兼容 Depth Anything / DINOv2 / Swin 等 backbone 的 hidden_sizes 列表配置
                if hasattr(cfg, "hidden_sizes") and isinstance(cfg.hidden_sizes, (list, tuple)):
                    return cfg.hidden_sizes[-1]
                # 3. 兼容常规网络属性
                for attr in ["hidden_size", "d_model", "embed_dim", "projection_dim", "num_features"]:
                    if hasattr(cfg, attr):
                        return getattr(cfg, attr)
            
            # 4. 如果 Config 中无法获取，尝试直接读取 Backbone 对象的属性
            if hasattr(self.backbone, "num_features"):
                return self.backbone.num_features
            if hasattr(self.backbone, "embed_dim"):
                return self.backbone.embed_dim

            return 768  # 兜底默认维度

    def forward(self, x):
        if not self.is_custom_classifier:
            # 由 AutoModelForImageClassification 或 timm 直接处理 forward 并输出 logits
            outputs = self.backbone(x)
            return outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            # 提取特征后送入自建的 classifier
            outputs = self.backbone(x)
            
            # --- 多层级/多结构特征提取兼容 ---
            feat = None
            if hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state
            elif hasattr(outputs, "feature_maps") and outputs.feature_maps:
                feat = outputs.feature_maps[-1]
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                feat = outputs.hidden_states[-1]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            elif isinstance(outputs, (tuple, list)):
                feat = outputs[-1]
            else:
                feat = outputs

            # 特殊情况处理：如果返回的是 dict 结构（如部分 SAM2 / 离散 Backbone）
            if isinstance(feat, dict):
                if "last_hidden_state" in feat:
                    feat = feat["last_hidden_state"]
                elif "vision_features" in feat:
                    feat = feat["vision_features"]
                else:
                    feat = list(feat.values())[-1]

            # --- 动态感知池化：彻底解决维度不匹配问题 ---
            if isinstance(feat, torch.Tensor):
                target_dim = self.classifier.in_features
                
                # 如果当前 Tensor 的最后一个维度正好与 classifier 要求的一致
                if feat.shape[-1] == target_dim:
                    if feat.dim() == 4:     # 例如 [B, H, W, C]
                        feat = feat.mean(dim=[1, 2])
                    elif feat.dim() == 3:   # 例如 [B, N, C]
                        feat = feat.mean(dim=1)
                # 如果当前 Tensor 的第2维 (C维度) 与 classifier 一致
                elif feat.dim() == 4 and feat.shape[1] == target_dim: # 例如 [B, C, H, W]
                    feat = feat.mean(dim=[-2, -1])
                else:
                    # 兜底适配：将除 Batch 以外的所有维度展平，如仍不匹配通过 Adaptive Avg Pooling 强行归一到 target_dim
                    if feat.dim() > 2:
                        # 查找哪个轴对应 target_dim
                        match_dims = [i for i, size in enumerate(feat.shape) if size == target_dim]
                        if match_dims:
                            # 保留 Batch 和匹配到的特征维度，对其他所有空间维度求均值
                            keep_dim = match_dims[0]
                            reduce_dims = [i for i in range(1, feat.dim()) if i != keep_dim]
                            feat = feat.mean(dim=reduce_dims)
                        else:
                            # 极罕见边界情况：展平后自适应池化
                            feat = feat.flatten(start_dim=1)
                            if feat.shape[-1] != target_dim:
                                feat = nn.functional.adaptive_avg_pool1d(feat.unsqueeze(1), target_dim).squeeze(1)

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
    feature_models = ["clip", "dino", "siglip", "depth-anything", "sam2", "mobileclip", "tinyvim", "internimage", "vit_base"]
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