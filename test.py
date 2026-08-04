import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from utils.dataset import CambridgeBridgeDataset
from models.builder import build_model_and_processor

def main():
    # 1. 读取全局配置与模型配置
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open("configs/vit/vit_base.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备进行测试: {device}")

    # 2. 定位保存的最佳模型路径
    # 默认读取最近一次生成的 checkpoints，你也可以手动填入具体文件夹路径
    exp_name = f"{model_cfg['model']['type']}_base_exp1"
    # 这里也可以替换为具体的日期文件夹，如 "outputs/2026-08-04/vit_base_exp1/checkpoints"
    ckpt_dir = os.path.join("outputs", "latest", exp_name, "checkpoints") 
    
    # 如果没找到 latest 链接，就提示输入或指定路径
    if not os.path.exists(ckpt_dir):
        # 寻找 outputs 目录下最新的实验输出
        outputs_root = "outputs"
        all_dates = sorted([d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))])
        if all_dates:
            latest_date = all_dates[-1]
            ckpt_dir = os.path.join(outputs_root, latest_date, exp_name, "checkpoints")

    print(f"即将从以下路径加载最佳模型权重: {ckpt_dir}")

    # 3. 准备数据集与测试集划分（必须使用与训练集完全相同的随机种子 42）
    _, processor = build_model_and_processor(model_cfg['model']['name'], num_classes=2, class_names=[])
    full_dataset = CambridgeBridgeDataset(base_cfg['data']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total_size = len(full_dataset)
    train_size = int(base_cfg['data']['train_split'] * total_size)
    val_size = int(base_cfg['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    # 使用相同的 manual_seed(42)，保证切分出的 test_dataset 与训练时完全一致！
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(base_cfg['seed'])
    )

    test_loader = DataLoader(test_dataset, batch_size=base_cfg['train']['batch_size'], shuffle=False)
    print(f"测试集准备就绪，共包含样本数: {len(test_dataset)}")

# 4. 加载微调好的模型
    if os.path.exists(os.path.join(ckpt_dir, "config.json")):
        # 如果是 HuggingFace ViT 格式
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            ckpt_dir,
            num_labels=num_classes,
            ignore_mismatched_sizes=True, # 防止分类头尺寸冲突
            id2label={i: name for i, name in enumerate(class_names)},
            label2id={name: i for i, name in enumerate(class_names)}
        ).to(device)
    else:
        # 如果是 PyTorch 通用 state_dict (.pth)
        model, _ = build_model_and_processor(model_cfg['model']['name'], num_classes, class_names)
        best_pth = os.path.join(ckpt_dir, "best.pth")
        model.load_state_dict(torch.load(best_pth, map_location=device))
        model.to(device)

    # 5. 执行测试评估
    model.eval()
    all_preds = []
    all_labels = []

    print("\n======== 开始测试集盲测评估 ========")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            # --- 修复核心：统一前向传播，优雅兼容 ViT 与 CNN ---
            if hasattr(model, "config") and "vit" in model.config.model_type.lower():
                outputs = model(pixel_values=images)
                logits = outputs.logits
            else:
                logits = model(images)
            
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # 6. 打印并输出科研论文标准评估报告
    print("\n" + "="*20 + " 最终论文评估结果 " + "="*20)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率 (Overall Accuracy): {acc * 100:.2f}%")
    print("="*58)

if __name__ == "__main__":
    main()