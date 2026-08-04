import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images) if hasattr(outputs := model(images), 'logits') else outputs
        # 兼容 HuggingFace 的 outputs.logits 和通用 PyTorch 的 outputs
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
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
            outputs = model(pixel_values=images) if hasattr(outputs := model(images), 'logits') else outputs
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(-1)
            corrects += torch.sum(preds == labels.data)
            
    return total_loss / total_samples, corrects.double() / total_samples