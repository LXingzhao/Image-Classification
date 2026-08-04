# utils/engine.py
import torch
from tqdm import tqdm

def extract_logits(output):
    """
    通用 logits 提取器：
    自动适配 PyTorch 原生 Tensor、timm 输出、HuggingFace ModelOutput/Dict 对象
    """
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "logits"):
        return output.logits
    elif isinstance(output, dict) and "logits" in output:
        return output["logits"]
    elif isinstance(output, (tuple, list)):
        return output[0]
    else:
        raise TypeError(f"无法从模型输出类型 {type(output)} 中提取 logits！")

def forward_step(model, images):
    """
    通用前向传播包装器：
    优先尝试 HuggingFace 的 pixel_values 传参，若失败则回退到标准 PyTorch 传参
    """
    try:
        if hasattr(model, "config"):
            output = model(pixel_values=images)
        else:
            output = model(images)
    except (TypeError, ValueError):
        # 兼容部分不需要 pixel_values 关键字的 HuggingFace/Custom 模型
        output = model(images)
        
    return extract_logits(output)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 获取通用 logits
        logits = forward_step(model, images)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(-1)
        corrects += torch.sum(preds == labels.data)
        
    return total_loss / total_samples, corrects.double() / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            # 获取通用 logits
            logits = forward_step(model, images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(-1)
            corrects += torch.sum(preds == labels.data)
            
    return total_loss / total_samples, corrects.double() / total_samples