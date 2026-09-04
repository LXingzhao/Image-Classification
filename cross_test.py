#cross_test.py
import os
import sys
import time
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.builder import build_model_and_processor


def get_dataset_class(dataset_type: str):
  dataset_type = dataset_type.lower()
  if dataset_type in ["sdnet2018", "sdnet"]:
    from utils.dataset import SDNET2018Dataset

    return SDNET2018Dataset
  else:
    raise ValueError(f"当前跨域测试仅支持 SDNET2018，未知类型: {dataset_type}")


# 24 个模型配置列表
MODELS = [
    "ResNet50",
    "ResNet152",
    "vit_base",
    "clip_vit_l14",
    "swin_v2_base",
    "convnext_large",
    "convnextv2_tiny",
    "efficientvit_m2",
    "efficientformerv2_s",
    "dinov2_base",
    "internimage_base",
    "mobilevitv3_small",
    "fastvit_sa12",
    "eva02_large",
    "starnet",
    "mobilenetv4_small",
    "repvit_m15",
    "mambavision_base",
    "yolo11_cls",
    "depth_anything_v2",
    "siglip2_base",
    "tinyvim_base",
    "sam2_hiera_base",
    "mobileclip2_b",
    "yolo26_cls",
]

# 新增 "D_P_W" 权重来源
SUBTYPES = ["D", "P", "W", "D_P_W"]
TARGET_SUBTYPES = ["D", "P", "W"]  # 测试目标仍为 D, P, W 三个子集
DATASET_NAME = "SDNET2018"


def find_best_ckpt(dataset_output_name, model_type):
  """自动查找对应子集下模型的 best.pth 路径"""
  exp_name = (
      f"{model_type}_exp1"
      if not model_type.endswith("_base")
      else f"{model_type[:-5]}_exp1"
  )
  outputs_root = os.path.join("outputs", dataset_output_name)

  if not os.path.exists(outputs_root):
    return None

  all_dates = sorted(
      [
          d
          for d in os.listdir(outputs_root)
          if os.path.isdir(os.path.join(outputs_root, d))
      ],
      reverse=True,
  )
  for date_dir in all_dates:
    target_path = os.path.join(
        outputs_root, date_dir, exp_name, "checkpoints", "best.pth"
    )
    if os.path.exists(target_path):
      return target_path
  return None


def load_merged_config(model_name, dataset_name, source_sub):
  """加载并合并配置"""
  cfg = {}
  with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  if "train" not in cfg:
    cfg["train"] = {}

  dataset_cfg_path = os.path.join(
      "configs", "datasets", f"{dataset_name}.yaml"
  )
  if os.path.exists(dataset_cfg_path):
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
      dataset_cfg = yaml.safe_load(f)
      cfg["dataset"] = dataset_cfg.get("dataset", {})
      if "train" in dataset_cfg and dataset_cfg["train"]:
        cfg["train"].update(dataset_cfg["train"])

  model_cfg_path = os.path.join(
      "configs", "models", dataset_name, source_sub, f"{model_name}.yaml"
  )
  if not os.path.exists(model_cfg_path):
    model_cfg_path = os.path.join("configs", "models", f"{model_name}.yaml")

  if os.path.exists(model_cfg_path):
    with open(model_cfg_path, "r", encoding="utf-8") as f:
      model_cfg = yaml.safe_load(f)
      cfg["model"] = model_cfg.get("model", {})
      if "train" in model_cfg and model_cfg["train"]:
        cfg["train"].update(model_cfg["train"])
  else:
    cfg["model"] = {"name": model_name, "type": model_name}

  return cfg


def evaluate_on_target(model, test_loader, device):
  """评估模型在特定测试集上的表现"""
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      outputs = model(images)
      logits = outputs.logits if hasattr(outputs, "logits") else outputs
      preds = logits.argmax(-1)

      all_preds.extend(preds.cpu().tolist())
      all_labels.extend(labels.tolist())

  acc = accuracy_score(all_labels, all_preds)
  precision, recall, f1, _ = precision_recall_fscore_support(
      all_labels, all_preds, average="macro", zero_division=0
  )
  return acc, precision, recall, f1


def save_txt_report(df, txt_file_path):
  """生成格式化的 TXT 评估报告"""
  with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(
        " SDNET2018 跨域/同域评估全景报告 (Generated:"
        f" {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
    )
    f.write("=" * 80 + "\n\n")

    # 1. 按模型聚合打印
    for model_name, group in df.groupby("模型名称"):
      f.write(f"【模型名称】: {model_name}\n")
      f.write("-" * 80 + "\n")
      f.write(
          f"{'权重来源(Source)':<15} {'测试目标(Target)':<15} {'测试类型':<12}"
          f" {'Accuracy (%)':<15} {'F1-Score':<10}\n"
      )
      f.write("-" * 80 + "\n")
      for _, row in group.iterrows():
        if row["权重来源(Source)"] == row["测试目标(Target)"]:
          test_type = "同域测试"
        elif row["权重来源(Source)"] == "D_P_W":
          test_type = "全域测试"
        else:
          test_type = "跨域测试"
        f.write(
            f"{row['权重来源(Source)']:<15} {row['测试目标(Target)']:<15}"
            f" {test_type:<12} {row['Accuracy (%)']:<15.2f}"
            f" {row['F1-Score']:<10.4f}\n"
        )
      f.write("\n" + "." * 80 + "\n\n")

    # 2. 矩阵格式对照表
    f.write("\n" + "=" * 80 + "\n")
    f.write(" 各模型 矩阵性能速查表 (Accuracy %)\n")
    f.write("=" * 80 + "\n")

    pivot_df = df.pivot(
        index="模型名称",
        columns=["权重来源(Source)", "测试目标(Target)"],
        values="Accuracy (%)",
    )
    f.write(pivot_df.to_string())
    f.write("\n\n" + "=" * 80 + "\n")


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("=" * 60)
  print(" 开始 SDNET2018 全模型 [同域 + 跨域 + 全域] 矩阵评估")
  print(f" 运行设备: {device}")
  print("=" * 60)

  summary_results = []

  for model_name in MODELS:
    print(f"\n>>>> 正在处理模型: {model_name} <<<<")

    for source_sub in SUBTYPES:
      if source_sub == "D_P_W":
        # 对应 outputs/SDNET2018/D_P_W
        source_dataset_output_name = os.path.join(DATASET_NAME, "D_P_W")
      else:
        # 对应 outputs/SDNET2018_D, outputs/SDNET2018_P, outputs/SDNET2018_W
        source_dataset_output_name = f"{DATASET_NAME}_{source_sub}"
      cfg = load_merged_config(model_name, DATASET_NAME, source_sub)
      model_type = cfg["model"]["type"]

      # 查找特定权重 (D, P, W 或 D_P_W)
      best_pth = find_best_ckpt(source_dataset_output_name, model_type)
      if not best_pth:
        print(
            f"  [跳过] 未找到 {model_name} 在【{source_dataset_output_name}】下的"
            " best.pth"
        )
        continue

      # 预加载与初始化模型
      try:
        _, processor = build_model_and_processor(
            cfg["model"]["name"], num_classes=2, class_names=[]
        )
        model, _ = build_model_and_processor(
            cfg["model"]["name"], num_classes=2, class_names=["0", "1"]
        )
        checkpoint = torch.load(best_pth, map_location=device)
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
      except Exception as e:
        print(
            f"  [错误] 加载模型/权重 {model_name} ({source_sub}) 失败: {e}"
        )
        continue

      # 评估 D, P, W 三个测试子集
      for target_sub in TARGET_SUBTYPES:
        DatasetClass = get_dataset_class(cfg["dataset"]["type"])
        target_dataset = DatasetClass(
            cfg["dataset"]["dir"], processor, sub_type=target_sub
        )

        total_size = len(target_dataset)
        train_size = int(cfg["dataset"]["train_split"] * total_size)
        val_size = int(cfg["dataset"]["val_split"] * total_size)
        test_size = total_size - train_size - val_size

        _, _, test_dataset = random_split(
            target_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )

        batch_size = int(cfg["train"].get("batch_size", 32))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        acc, precision, recall, f1 = evaluate_on_target(
            model, test_loader, device
        )

        if source_sub == target_sub:
          tag = "同域"
        elif source_sub == "D_P_W":
          tag = "全域"
        else:
          tag = "跨域"

        print(
            f"  ↳ [{source_sub}权重 -> {target_sub}测试集] ({tag}) | Acc:"
            f" {acc*100:.2f}% | F1: {f1:.4f}"
        )

        summary_results.append({
            "模型名称": model_name,
            "权重来源(Source)": source_sub,
            "测试目标(Target)": target_sub,
            "测试集样本数": len(test_dataset),
            "Accuracy (%)": round(acc * 100, 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "权重路径": best_pth,
        })

  # 导出结果文件为“模型详情”
  if summary_results:
    target_dir = os.path.join("outputs", "SDNET2018")
    os.makedirs(target_dir, exist_ok=True)

    txt_save_path = os.path.join(target_dir, "模型详情.txt")
    csv_save_path = os.path.join(target_dir, "模型详情.csv")

    df = pd.DataFrame(summary_results)
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    save_txt_report(df, txt_save_path)

    print("\n" + "=" * 60)
    print("[完成] 所有矩阵评估任务结束！")
    print(f" TXT 结果文件已保存至: {txt_save_path}")
    print(f" CSV 结果文件已保存至: {csv_save_path}")
    print("=" * 60)
  else:
    print(
        "\n[警告] 未检测到有效的权重文件，请确保模型已成功训练。"
    )


if __name__ == "__main__":
  main()