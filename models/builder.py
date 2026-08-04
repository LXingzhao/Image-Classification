# utils/engine.py
import torch
from tqdm import tqdm

def get_logits(model, images):
    """辅助函数：优雅适配 HuggingFace 和 timm/torchvision 模型"""
    if hasattr(model, "config") and "vit" in model.config.model_type.lower():
        outputs = model(pixel_values=images)
        return outputs.logits
    else:
        return model(images)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = get_logits(model, images)
        
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
            logits = get_logits(model, images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(-1)
            corrects += torch.sum(preds == labels.data)
            
    return total_loss / total_samples, corrects.double() / total_samples