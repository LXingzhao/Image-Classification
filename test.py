#test.py
import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from utils.dataset import CambridgeBridgeDataset
from models.builder import build_model_and_processor

def main():
    # 0. 命令行参数解析
    parser = argparse.ArgumentParser(description="评估模型在测试集上的表现")
    parser.add_argument("--config", type=str, required=True, help="模型配置文件路径")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="手动指定权重保存目录路径")
    args = parser.parse_args()

    # 1. 读取全局配置与模型配置
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    with open(args.config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备进行测试: {device}")

    # 2. 定位保存的最佳模型路径
    exp_name = f"{model_cfg['model']['type']}_base_exp1"
    
    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = os.path.join("outputs", "latest", exp_name, "checkpoints") 
        if not os.path.exists(ckpt_dir):
            outputs_root = "outputs"
            if os.path.exists(outputs_root):
                all_dates = sorted([d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))], reverse=True)
                for date_dir in all_dates:
                    target_path = os.path.join(outputs_root, date_dir, exp_name, "checkpoints")
                    if os.path.exists(target_path):
                        ckpt_dir = target_path
                        break

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"\n[错误] 未找到模型【{model_cfg['model']['type']}】的权重目录: {ckpt_dir}\n"
            f"请先运行训练命令: python train.py --config {args.config}"
        )

    print(f"即将从以下路径加载最佳模型权重: {ckpt_dir}")

    # 3. 准备数据集与测试集划分
    _, processor = build_model_and_processor(model_cfg['model']['name'], num_classes=2, class_names=[])
    full_dataset = CambridgeBridgeDataset(base_cfg['data']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total_size = len(full_dataset)
    train_size = int(base_cfg['data']['train_split'] * total_size)
    val_size = int(base_cfg['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(base_cfg['seed'])
    )

    test_loader = DataLoader(test_dataset, batch_size=base_cfg['train']['batch_size'], shuffle=False)
    print(f"测试集准备就绪，共包含样本数: {len(test_dataset)}")

# 4. 实例化模型并加载权重
    model, _ = build_model_and_processor(model_cfg['model']['name'], num_classes, class_names)
    best_pth = os.path.join(ckpt_dir, "best.pth")
    
    if os.path.exists(best_pth):
        checkpoint = torch.load(best_pth, map_location=device)
        
        # 自动兼容字典封装格式
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # 核心改动：设置 strict=False，忽略未微调的 Backbone 缺失层
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"成功加载本地权重文件: {best_pth}")
        
        if missing_keys:
            print(f"[提示] 有 {len(missing_keys)} 个预训练层未在权重文件中找到（已自动保留默认预训练参数）")
    else:
        raise FileNotFoundError(f"在路径 {ckpt_dir} 下未找到权重文件 best.pth")
        
    model.to(device)

    # 5. 执行评估
    model.eval()
    all_preds = []
    all_labels = []

    print("\n======== 开始测试集盲测评估 ========")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            outputs = model(images)
            
            # 通用 logits 提取逻辑
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # 6. 打印评估报告
    print("\n" + "="*20 + " 最终评估结果 " + "="*20)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率 (Overall Accuracy): {acc * 100:.2f}%")
    print("="*58)

if __name__ == "__main__":
    main()