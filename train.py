import os
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
    # 1. 加载配置文件
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    # with open("configs/vit/vit_base.yaml", "r", encoding="utf-8") as f:     # ViT-Base  配置文件
    #     model_cfg = yaml.safe_load(f)
    with open("configs/cnn/resnet50.yaml", "r", encoding="utf-8") as f:       # ResNet50  配置文件
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 准备输出路径 outputs/2026-08-04/vit_base_exp1/
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_name = f"{model_cfg['model']['type']}_base_exp1"
    save_dir = os.path.join("outputs", today_date, exp_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3. 初始化模型与处理器
    # 先预置 processor 临时读取 Dataset class 结构
    _, processor = build_model_and_processor(model_cfg['model']['name'], num_classes=2, class_names=[])
    full_dataset = CambridgeBridgeDataset(base_cfg['data']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"成功载入数据集！包含类别: {class_names}，总样本数: {len(full_dataset)}")

    model, processor = build_model_and_processor(model_cfg['model']['name'], num_classes, class_names)
    model.to(device)

    # 4. 划分数据集
    total_size = len(full_dataset)
    train_size = int(base_cfg['data']['train_split'] * total_size)
    val_size = int(base_cfg['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(base_cfg['seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=base_cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=base_cfg['train']['batch_size'], shuffle=False)

    # 5. 优化器与损失函数
    optimizer = AdamW(model.parameters(), lr=float(base_cfg['train']['lr']), weight_decay=base_cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 6. 循环训练
    for epoch in range(base_cfg['train']['epochs']):
        print(f"\n======== Epoch {epoch + 1}/{base_cfg['train']['epochs']} ========")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型权重到 outputs/ 目录下
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
            print(f"-> 已保存当前最佳模型到 {ckpt_dir}")

if __name__ == "__main__":
    main()