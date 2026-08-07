# train.py
import os
import argparse
import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from utils.dataset import CambridgeBridgeDataset
from models.builder import build_model_and_processor
from utils.engine import train_one_epoch, evaluate

def main():
    # 1. 支持命令行传参切换模型配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml", 
                        help="模型配置文件路径")
    args = parser.parse_args()

    # 2. 加载配置文件并进行深度合并
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # 如果模型专属 yaml 中覆盖了 train 配置（例如单独设置了更小的 batch_size），进行覆盖更新
    if "train" in model_cfg:
        cfg["train"].update(model_cfg["train"])
    cfg["model"] = model_cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} | 加载模型配置: {args.config}")
    print(f"当前 Batch Size: {cfg['train']['batch_size']}")

    # 3. 准备输出路径
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_name = f"{cfg['model']['type']}_base_exp1"
    save_dir = os.path.join("outputs", today_date, exp_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4. 初始化模型与处理器
    _, processor = build_model_and_processor(cfg['model']['name'], num_classes=2, class_names=[])
    full_dataset = CambridgeBridgeDataset(cfg['data']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"成功载入数据集！包含类别: {class_names}，总样本数: {len(full_dataset)}")

    model, processor = build_model_and_processor(cfg['model']['name'], num_classes, class_names)
    model.to(device)

    # 5. 划分数据集
    total_size = len(full_dataset)
    train_size = int(cfg['data']['train_split'] * total_size)
    val_size = int(cfg['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )

    # 使用从配置解析出的 batch_size
    batch_size = cfg['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 6. 全模型通用学习率自适应匹配
    base_lr = float(cfg['train']['lr'])
    model_name_lower = cfg['model']['name'].lower()
    
    if any(k in model_name_lower for k in ["clip", "dino", "depth-anything", "eva02", "large", "sam2"]):
        # 大参数量预训练模型或特征提取器，采用极低学习率保护特征
        lr = 1e-5
    elif any(k in model_name_lower for k in ["resnet", "convnext", "mobilenet"]):
        # 卷积为主的模型，采用相对较高的学习率
        lr = base_lr * 2
    else:
        lr = base_lr

    print(f"当前模型学习率 (Learning Rate) 设置为: {lr}")

    # 过滤掉 requires_grad=False 的冻结参数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=cfg['train']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # 初始化混合精度 Scaler (防溢出)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0

    # 7. 循环训练
    for epoch in range(cfg['train']['epochs']):
        print(f"\n======== Epoch {epoch + 1}/{cfg['train']['epochs']} ========")
        
        # 将 scaler 传入 engine 的 train_one_epoch
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        except TypeError:
            # 如果 utils/engine.py 暂不支持 scaler 参数，退回普通调用
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(ckpt_dir, "best.pth")
            
            # 1. 解包原生模型（防止 DDP/多卡影响）
            raw_model = model.module if hasattr(model, 'module') else model
            
            # 2. 以标准字典格式保存 checkpoint
            torch.save({
                'model': raw_model.state_dict(),
                'val_acc': best_val_acc,
                'epoch': epoch + 1
            }, best_model_path)
            
            # 3. 将 HuggingFace 格式文件隔离保存到独立子目录，防止污染或覆盖 best.pth
            hf_save_dir = os.path.join(ckpt_dir, "hf_format")
            if hasattr(raw_model, 'save_pretrained'):
                raw_model.save_pretrained(hf_save_dir)
            if hasattr(processor, 'save_pretrained'):
                processor.save_pretrained(hf_save_dir)
                
            print(f"-> 已保存当前最佳模型到 {ckpt_dir} (Best Val Acc: {best_val_acc:.4f})")

if __name__ == "__main__":
    main()