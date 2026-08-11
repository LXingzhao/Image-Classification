# test.py
import os
import sys
import time
import yaml
import torch
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from models.builder import build_model_and_processor

def get_dataset_class(dataset_type: str):
    dataset_type = dataset_type.lower()
    if dataset_type == "cambridge":
        from utils.dataset import CambridgeBridgeDataset
        return CambridgeBridgeDataset
    elif dataset_type in ["gyu_det", "gyu"]:
        from utils.dataset import GYUDETDataset
        return GYUDETDataset
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")


class AppendLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

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
    # 0. 命令行参数解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="评估模型在测试集上的表现")
    parser.add_argument("--model", type=str, default="dinov2_base", 
                        help="模型配置文件名称 (如 dinov2_base 或 vit_base)")
    parser.add_argument("--dataset", type=str, default="CambridgeBridge", 
                        help="数据集配置文件名称 (如 CambridgeBridge 或 gyu_det)")
    parser.add_argument("--config", type=str, default="", help="手动指定模型配置文件全路径")
    parser.add_argument("--dataset_config", type=str, default="", help="手动指定数据集配置文件全路径")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="手动指定权重保存目录路径")
    args = parser.parse_args()

    # 优先补全数据集配置文件路径 (configs/datasets/CambridgeBridge.yaml)
    if not args.dataset_config:
        dataset_filename = args.dataset if args.dataset.endswith(".yaml") else f"{args.dataset}.yaml"
        args.dataset_config = os.path.join("configs", "datasets", dataset_filename)

    # 自动匹配数据集子目录下的模型配置文件 (configs/models/CambridgeBridge/vit_base.yaml)
    if not args.config:
        dataset_name_from_path = os.path.splitext(os.path.basename(args.dataset_config))[0]
        model_filename = args.model if args.model.endswith(".yaml") else f"{args.model}.yaml"
        
        # 优先寻找数据集专属模型配置
        args.config = os.path.join("configs", "models", dataset_name_from_path, model_filename)
        
        # 降级备选：如果专属配置不存在，则尝试读取全局通用模型配置 (configs/models/vit_base.yaml)
        if not os.path.exists(args.config):
            fallback_path = os.path.join("configs", "models", model_filename)
            if os.path.exists(fallback_path):
                args.config = fallback_path

    # ---------------------------------------------------------
    # 1. 读取并深度合并配置
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 2. 定位按数据集隔离的权重与日志路径
    # ---------------------------------------------------------
    dataset_name = cfg['dataset']['name']
    model_type = cfg['model']['type']
    exp_name = f"{model_type}_exp1" if not model_type.endswith("_base") else f"{model_type[:-5]}_exp1"
    
    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    else:
        outputs_dataset_root = os.path.join("outputs", dataset_name)
        ckpt_dir = ""
        if os.path.exists(outputs_dataset_root):
            all_dates = sorted([d for d in os.listdir(outputs_dataset_root) if os.path.isdir(os.path.join(outputs_dataset_root, d))], reverse=True)
            for date_dir in all_dates:
                target_path = os.path.join(outputs_dataset_root, date_dir, exp_name, "checkpoints")
                if os.path.exists(target_path):
                    ckpt_dir = target_path
                    break

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"\n[错误] 未在数据集【{dataset_name}】路径下找到模型【{cfg['model']['type']}】的权重目录: {ckpt_dir}\n"
            f"请先运行训练命令: python train.py --model {args.model} --dataset {args.dataset}"
        )

    # 重定向测试日志：自动匹配该文件夹下的 _log.txt 日志文件
    save_dir = os.path.dirname(ckpt_dir)
    log_file_path = None

    # 寻找以 _log.txt 结尾的文件，或者默认的 train_log.txt
    for file in os.listdir(save_dir):
        if file.endswith("_log.txt") or file == "train_log.txt":
            log_file_path = os.path.join(save_dir, file)
            break

    if log_file_path and os.path.exists(log_file_path):
        sys.stdout = AppendLogger(log_file_path)
        print(f"[日志记录] 已将测试日志重定向追加至: {log_file_path}")
    else:
        print("[提示] 未匹配到已有训练日志文件，测试打印将仅在控制台显示。")

    print("\n" + "="*20 + " 开始测试集评估 " + "="*20)
    print(f"评估数据集: {dataset_name}")
    print(f"正在使用设备进行测试: {device}")
    print(f"即将从以下路径加载最佳模型权重: {ckpt_dir}")

    # ---------------------------------------------------------
    # 3. 动态加载数据集与划分测试集
    # ---------------------------------------------------------
    _, processor = build_model_and_processor(cfg['model']['name'], num_classes=2, class_names=[])
    
    DatasetClass = get_dataset_class(cfg['dataset']['type'])
    full_dataset = DatasetClass(cfg['dataset']['dir'], processor)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total_size = len(full_dataset)
    train_size = int(cfg['dataset']['train_split'] * total_size)
    val_size = int(cfg['dataset']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )

    batch_size = int(cfg['train']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"测试集准备就绪，包含类别: {class_names}，测试集样本数: {len(test_dataset)}")

    # ---------------------------------------------------------
    # 4. 实例化模型并加载权重
    # ---------------------------------------------------------
    model, _ = build_model_and_processor(cfg['model']['name'], num_classes, class_names)
    best_pth = os.path.join(ckpt_dir, "best.pth")
    
    if os.path.exists(best_pth):
        checkpoint = torch.load(best_pth, map_location=device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"成功加载本地权重文件: {best_pth}")
        
        if missing_keys:
            print(f"[提示] 有 {len(missing_keys)} 个预训练层未在权重文件中找到（已自动保留默认预训练参数）")
    else:
        raise FileNotFoundError(f"在路径 {ckpt_dir} 下未找到权重文件 best.pth")
        
    model.to(device)

    # ---------------------------------------------------------
    # 5. 执行评估（统计推理时间与保存预测概率）
    # ---------------------------------------------------------
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n======== 开始测试集盲测评估 ========")
    test_start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    test_total_time = time.time() - test_start_time

    # 保存测试结果字典用于后续绘图 (混淆矩阵 / ROC / PR)
    test_results = {
        "class_names": class_names,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs
    }
    results_save_path = os.path.join(save_dir, "test_results.pth")
    torch.save(test_results, results_save_path)
    print(f"\n[保存成功] 测试集评估细节已保存至: {results_save_path}")

    # ---------------------------------------------------------
    # 6. 打印评估报告
    # ---------------------------------------------------------
    print("\n" + "="*20 + " 最终评估结果 " + "="*20)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率 (Overall Accuracy): {acc * 100:.2f}%")
    print(f"测试集推理评估总耗时 (Test Total Time): {test_total_time:.2f}s")
    print("="*58)

if __name__ == "__main__":
    main()