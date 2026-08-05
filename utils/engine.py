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
    支持 DDP / DataParallel 包装后的模型检查
    """
    # 检查原始模型或 DDP/DP 包装后的内部模型是否有 config
    unwrap_model = getattr(model, "module", model)
    
    try:
        if hasattr(unwrap_model, "config"):
            output = model(pixel_values=images)
        else:
            output = model(images)
    except (TypeError, ValueError):
        # 兼容部分不需要 pixel_values 关键字的 HuggingFace/Custom 模型
        output = model(images)
        
    return extract_logits(output)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    支持 AMP 混合精度训练的一轮训练函数
    """
    model.train()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 混合精度前向传播 (AMP)
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = forward_step(model, images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_step(model, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(-1)
        correct_count = (preds == labels).sum().item()
        corrects += correct_count
        
        # 动态更新 tqdm 实时指标
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct_count / batch_size:.4f}"
        })
        
    epoch_loss = total_loss / total_samples
    epoch_acc = corrects / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    评估函数
    """
    model.eval()
    total_loss, corrects = 0.0, 0
    total_samples = len(dataloader.dataset)
    
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 如果支持硬件，验证阶段也可以开启 autocast 提速
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = forward_step(model, images)
                    loss = criterion(logits, labels)
            else:
                logits = forward_step(model, images)
                loss = criterion(logits, labels)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(-1)
            corrects += (preds == labels).sum().item()
            
    epoch_loss = total_loss / total_samples
    epoch_acc = corrects / total_samples
    return epoch_loss, epoch_acc