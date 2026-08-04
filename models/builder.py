import timm
from transformers import ViTForImageClassification, ViTImageProcessor

def build_model_and_processor(model_name: str, num_classes: int, class_names: list):
    """
    统一模型构建入口：自动兼容 HuggingFace 模型与 timm 模型
    """
    if "google/vit" in model_name:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label={i: name for i, name in enumerate(class_names)},
            label2id={name: i for i, name in enumerate(class_names)}
        )
    else:
        # 未来扩展：给 ResNet50 / ConvNeXt 等 timm 模型准备
        processor = None
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
    return model, processor