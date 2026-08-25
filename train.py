# train.py
import os
import sys
import time
import argparse
import datetime
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

def get_dataset_class(dataset_type: str):
    dataset_type = dataset_type.lower()
    if dataset_type == "cambridge":
        from utils.dataset import CambridgeBridgeDataset
        return CambridgeBridgeDataset
    elif dataset_type in ["gyu_det", "gyu"]:
        from utils.dataset import GYUDETDataset
        return GYUDETDataset
    elif dataset_type in ["sdnet2018", "sdnet"]:
        from utils.dataset import SDNET2018Dataset
        return SDNET2018Dataset
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")


class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
        abs_path = os.path.abspath(filepath)
        folder_path = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        
        header_info = f"保存路径: {folder_path} | 文件名: {file_name}\n" + "="*60 + "\n"
        self.terminal.write(header_info)
        self.log.write(header_info)
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()


def main():
    # ---------------------------------------------------------
    # 1. 命令行参数配置（兼容 SDNET2018/D 子类型路径）
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2_base", 
                        help="模型配置文件名称 (如 dinov2_base 或 vit_base)")
    parser.add_argument("--dataset", type=str, default="SDNET2018", 
                        help="数据集名称或路径 (如 SDNET2018, SDNET2018/D, CambridgeBridge)")
    parser.add_argument("--sub_type", type=str, default="D", 
                        help="子数据集类型，针对 SDNET2018 可选: D, P, W, ALL")
    parser.add_argument("--config", type=str, default="", 
                        help="手动指定模型配置文件全路径")
    parser.add_argument("--dataset_config", type=str, default="", 
                        help="手动指定数据集配置文件全路径")
    parser.add_argument("--patience", type=int, default=10, 
                        help="早停机制容忍轮数")
    args = parser.parse_args()

    # 解析 dataset 与 sub_type（兼容 --dataset SDNET2018/D 格式）
    dataset_input = args.dataset.replace("\\", "/")
    sub_type = args.sub_type

    if "/" in dataset_input:
        parts = dataset_input.split("/")
        dataset_base_name = parts[0]
        sub_type = parts[1]
    else:
        dataset_base_name = dataset_input

    # 优先补全数据集配置文件路径 (configs/datasets/SDNET2018.yaml)
    if not args.dataset_config:
        dataset_filename = dataset_base_name if dataset_base_name.endswith(".yaml") else f"{dataset_base_name}.yaml"
        args.dataset_config = os.path.join("configs", "datasets", dataset_filename)

    # 自动匹配模型配置文件：优先找 configs/models/SDNET2018/D/vit_base.yaml
    if not args.config:
        dataset_name_from_path = os.path.splitext(os.path.basename(args.dataset_config))[0]
        model_filename = args.model if args.model.endswith(".yaml") else f"{args.model}.yaml"
        
        # 路径策略 1: configs/models/SDNET2018/D/vit_base.yaml
        path_with_subtype = os.path.join("configs", "models", dataset_name_from_path, sub_type, model_filename) if sub_type else ""
        # 路径策略 2: configs/models/SDNET2018/vit_base.yaml
        path_with_dataset = os.path.join("configs", "models", dataset_name_from_path, model_filename)
        # 路径策略 3: configs/models/vit_base.yaml
        fallback_path = os.path.join("configs", "models", model_filename)

        if path_with_subtype and os.path.exists(path_with_subtype):
            args.config = path_with_subtype
        elif os.path.exists(path_with_dataset):
            args.config = path_with_dataset
        elif os.path.exists(fallback_path):
            args.config = fallback_path
        else:
            args.config = path_with_subtype if path_with_subtype else path_with_dataset

    # ---------------------------------------------------------
    # 2. 读取并合并配置
    # ---------------------------------------------------------
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "train" not in cfg:
        cfg["train"] = {}

    with open(args.dataset_config, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
        cfg["dataset"] = dataset_cfg.get("dataset", {})
        if "train" in dataset_cfg and dataset_cfg["train"]:
            cfg["train"].update(dataset_cfg["train"])

    with open(args.config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
        cfg["model"] = model_cfg.get("model", {})
        if "train" in model_cfg and model_cfg["train"]:
            cfg["train"].update(model_cfg["train"])

# ---------------------------------------------------------
    # 3. 构造输出路径：outputs/SDNET2018/D/{YYYY-MM-DD}/{model_type}_exp1
    # ---------------------------------------------------------
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 提取数据集名称与子类型，构建多级输出文件夹
    dataset_name = cfg['dataset']['name']
    
    if sub_type:
        save_dir = os.path.join("outputs", dataset_name, sub_type, today_date, exp_name)
        log_prefix = f"{dataset_name}_{sub_type}"
    else:
        save_dir = os.path.join("outputs", dataset_name, today_date, exp_name)
        log_prefix = dataset_name
    
    model_type = cfg['model']['type']
    exp_name = f"{model_type}_exp1" if not model_type.endswith("_base") else f"{model_type[:-5]}_exp1"
    
    save_dir = os.path.join("outputs", dataset_name, sub_type, today_date, exp_name) if sub_type else os.path.join("outputs", dataset_name, today_date, exp_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_filename = f"{log_prefix}_{model_type}_log.txt"
    log_file_path = os.path.join(save_dir, log_filename)
    sys.stdout = Logger(log_file_path)

    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb_logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = float(cfg['train']['lr'])
    batch_size = int(cfg['train']['batch_size'])
    weight_decay = float(cfg['train']['weight_decay'])

    print(f"正在使用设备: {device}")
    print(f"加载模型配置: {args.config} | 数据集配置: {args.dataset_config}")
    print("================ 超参数与数据集配置 ================")
    print(f"数据集名称 (Dataset Name) : {cfg['dataset']['name']}")
    print(f"数据集子类型 (Sub Type)    : {sub_type}")
    print(f"数据集路径 (Data Dir)     : {cfg['dataset']['dir']}")
    print(f"模型名称 (Model Type)     : {cfg['model']['type']}")
    print(f"模型ID (Model ID)         : {cfg['model']['name']}")
    print(f"批次大小 (Batch Size)     : {cfg['train']['batch_size']}")
    print(f"学习率 (Learning Rate)    : {cfg['train']['lr']}")
    print(f"权重衰减 (Weight Decay)   : {cfg['train']['weight_decay']}")
    print(f"输出保存路径              : {save_dir}")
    print("===================================================")

    # ---------------------------------------------------------
    # 4. 构建数据与模型
    # ---------------------------------------------------------
    from models.builder import build_model_and_processor

    _, processor = build_model_and_processor(cfg['model']['name'], num_classes=2, class_names=[])
    
    DatasetClass = get_dataset_class(cfg['dataset']['type'])
    
    # 针对 SDNET2018 传入 sub_type 参数
    if DatasetClass.__name__ == "SDNET2018Dataset":
        full_dataset = DatasetClass(cfg['dataset']['dir'], processor, sub_type=sub_type)
    else:
        full_dataset = DatasetClass(cfg['dataset']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"成功载入数据集 [{sub_type}]！包含类别: {class_names}，总样本数: {len(full_dataset)}")

    model, processor = build_model_and_processor(cfg['model']['name'], num_classes, class_names)
    model.to(device)

    # ---------------------------------------------------------
    # 5. 划分数据集与 DataLoader
    # ---------------------------------------------------------
    total_size = len(full_dataset)
    train_size = int(cfg['dataset']['train_split'] * total_size)
    val_size = int(cfg['dataset']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------------------------------------------------------
    # 6. 训练准备与主循环
    # ---------------------------------------------------------
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_epoch = 0
    best_metrics = {}
    patience = args.patience
    patience_counter = 0

    total_train_start_time = time.time()
    completed_epochs = 0
    max_epochs = cfg['train']['epochs']

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }

    from utils.engine import train_one_epoch, evaluate

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        print(f"\n======== Epoch {epoch + 1}/{max_epochs} ========")
        
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        except TypeError:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        completed_epochs += 1

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_time"].append(float(epoch_time))

        tb_writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        tb_writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
        tb_writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"本轮训练耗时: {epoch_time:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_metrics = {
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc
            }
            patience_counter = 0
            best_model_path = os.path.join(ckpt_dir, "best.pth")
            
            raw_model = model.module if hasattr(model, 'module') else model
            
            torch.save({
                'model': raw_model.state_dict(),
                'val_acc': best_val_acc,
                'epoch': best_epoch
            }, best_model_path)
            
            hf_save_dir = os.path.join(ckpt_dir, "hf_format")
            if hasattr(raw_model, 'save_pretrained'):
                raw_model.save_pretrained(hf_save_dir)
            if hasattr(processor, 'save_pretrained'):
                processor.save_pretrained(hf_save_dir)
                
            print(f"-> 验证集准确率提升至 {best_val_acc:.4f}！最佳模型已保存至 {ckpt_dir}")
        else:
            patience_counter += 1
            print(f"-> 验证集准确率未提升 ({val_acc:.4f} <= {best_val_acc:.4f})，早停计数器 Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] 连续 {patience} 轮未提升，停止训练！")
            break

    history_json_path = os.path.join(save_dir, "history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    tb_writer.close()

    total_time = time.time() - total_train_start_time
    avg_epoch_time = total_time / completed_epochs if completed_epochs > 0 else 0.0

    print("\n" + "="*20 + " 训练总结报告 " + "="*20)
    print(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)} | 测试集样本数: {len(test_dataset)}")
    print(f"实际训练总轮数: {completed_epochs}/{max_epochs}")
    print(f"最佳模型出现轮数: Epoch {best_epoch}")
    if best_metrics:
        print(f"最佳轮次指标数据:")
        print(f"  - Train Loss: {best_metrics['train_loss']:.4f} | Train Acc: {best_metrics['train_acc']:.4f}")
        print(f"  - Val Loss  : {best_metrics['val_loss']:.4f} | Val Acc  : {best_metrics['val_acc']:.4f}")
    print(f"训练历史指标已写入: {history_json_path}")
    print(f"训练总耗时: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"每轮平均耗时: {avg_epoch_time:.2f}s")
    print("="*58)

if __name__ == "__main__":
    main()