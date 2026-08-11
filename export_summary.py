# export_summary.py
import os
import re
import argparse
import pandas as pd

def parse_log_file(log_path):
    """解析单个 log.txt 文件并提取字段"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}
    
    # 0. 提取路径中的实验元信息
    # 路径示例: outputs/CambridgeBridge/2026-08-11/vit_exp1/CambridgeBridge_vit_base_log.txt
    normalized_path = log_path.replace('\\', '/')
    path_parts = normalized_path.split('/')
    if len(path_parts) >= 4:
        data["实验日期"] = path_parts[-3]
        data["实验名称"] = path_parts[-2]

    # 1. 提取配置与超参数
    data["数据集"] = re.search(r"数据集 \(Dataset Name\):\s*([^\n]+)", content).group(1).strip() if re.search(r"数据集 \(Dataset Name\):\s*([^\n]+)", content) else ""
    data["模型权重/名称"] = re.search(r"模型名称 \(Model\):\s*([^\n]+)", content).group(1).strip() if re.search(r"模型名称 \(Model\):\s*([^\n]+)", content) else ""
    
    batch_size = re.search(r"批次大小 \(Batch Size\):\s*(\d+)", content)
    data["Batch Size"] = int(batch_size.group(1)) if batch_size else None
    
    lr = re.search(r"学习率 \(Learning Rate\):\s*([^\n]+)", content)
    data["Learning Rate"] = float(lr.group(1)) if lr else None
    
    weight_decay = re.search(r"权重衰减 \(Weight Decay\):\s*([^\n]+)", content)
    data["Weight Decay"] = float(weight_decay.group(1)) if weight_decay else None

    # 2. 提取训练总结
    train_epochs = re.search(r"实际训练总轮数:\s*(\d+)/(\d+)", content)
    if train_epochs:
        data["实际训练轮数"] = int(train_epochs.group(1))
        data["设定最大轮数"] = int(train_epochs.group(2))
    
    best_epoch = re.search(r"最佳模型出现轮数:\s*Epoch\s*(\d+)", content)
    data["最佳Epoch"] = int(best_epoch.group(1)) if best_epoch else None
    
    best_train_loss = re.search(r"Train Loss:\s*([\d\.]+)\s*\|\s*Train Acc:\s*([\d\.]+)", content)
    if best_train_loss:
        data["最佳Train Loss"] = float(best_train_loss.group(1))
        data["最佳Train Acc"] = float(best_train_loss.group(2))

    best_val_loss = re.search(r"Val Loss\s*:\s*([\d\.]+)\s*\|\s*Val Acc\s*:\s*([\d\.]+)", content)
    if best_val_loss:
        data["最佳Val Loss"] = float(best_val_loss.group(1))
        data["最佳Val Acc"] = float(best_val_loss.group(2))

    train_time = re.search(r"训练总耗时:\s*([\d\.]+)s", content)
    data["训练总耗时(s)"] = float(train_time.group(1)) if train_time else None
    
    avg_train_time = re.search(r"每轮平均耗时:\s*([\d\.]+)s", content)
    data["单轮平均耗时(s)"] = float(avg_train_time.group(1)) if avg_train_time else None

    # 3. 提取测试集评估指标
    test_acc = re.search(r"测试集准确率 \(Overall Accuracy\):\s*([\d\.]+)%", content)
    data["Test Acc (%)"] = float(test_acc.group(1)) if test_acc else None

    test_samples = re.search(r"测试集样本数:\s*(\d+)", content)
    num_test_samples = int(test_samples.group(1)) if test_samples else None

    test_time = re.search(r"测试集推理评估总耗时 \(Test Total Time\):\s*([\d\.]+)s", content)
    if test_time:
        total_time_s = float(test_time.group(1))
        data["测试推理总耗时(s)"] = total_time_s
        if num_test_samples:
            data["单样本推理延迟(ms/img)"] = round((total_time_s / num_test_samples) * 1000, 2)

    # 4. 解析 classification_report (全局 average 与细分类别 F1)
    macro_avg = re.search(r"macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)", content)
    if macro_avg:
        data["Macro Precision"] = float(macro_avg.group(1))
        data["Macro Recall"] = float(macro_avg.group(2))
        data["Macro F1"] = float(macro_avg.group(3))

    weighted_avg = re.search(r"weighted avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)", content)
    if weighted_avg:
        data["Weighted Precision"] = float(weighted_avg.group(1))
        data["Weighted Recall"] = float(weighted_avg.group(2))
        data["Weighted F1"] = float(weighted_avg.group(3))

    # 提取各个特定类别的 F1-Score
    class_lines = re.findall(r"^\s*([A-Za-z0-9_\-\s]+?)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)\s*$", content, re.MULTILINE)
    for cls_name, prec, rec, f1, supp in class_lines:
        cls_name = cls_name.strip()
        if cls_name not in ["accuracy", "macro avg", "weighted avg"]:
            data[f"{cls_name} (F1)"] = float(f1)

    return data


def main():
    parser = argparse.ArgumentParser(description="自动汇总相同数据集下的模型评估结果至 Excel")
    parser.add_argument("--dataset", type=str, default="CambridgeBridge", help="目标数据集名称")
    args = parser.parse_args()

    dataset_dir = os.path.join("outputs", args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"[错误] 数据集目录不存在: {dataset_dir}")
        return

    # 递归搜索所有日志文件
    log_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("_log.txt") or file == "train_log.txt":
                log_files.append(os.path.join(root, file))

    if not log_files:
        print(f"[提示] 在 {dataset_dir} 路径下未查找到任何日志文件！")
        return

    print(f"搜寻到 {len(log_files)} 个日志文件，正在解析汇总...")
    
    all_data = []
    for log_path in log_files:
        try:
            parsed_data = parse_log_file(log_path)
            if parsed_data:
                all_data.append(parsed_data)
        except Exception as e:
            print(f"[警告] 解析日志失败 {log_path}: {e}")

    df = pd.DataFrame(all_data)

    # 优先展示的核心列排序
    first_cols = [
        "实验名称", "实验日期", "数据集", "模型权重/名称", "Test Acc (%)", 
        "Macro F1", "Weighted F1", "单样本推理延迟(ms/img)", 
        "Learning Rate", "Batch Size", "最佳Epoch", "实际训练轮数"
    ]
    
    existing_first_cols = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_first_cols]
    df = df[existing_first_cols + other_cols]

    # 导出至 Excel
    output_excel_path = os.path.join(dataset_dir, f"{args.dataset}_summary.xlsx")
    
    # 使用 openpyxl 引擎自动调整列宽
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="模型汇总指标对比", index=False)
        
        # 美化表格：根据数据长度设置列宽
        worksheet = writer.sheets["模型汇总指标对比"]
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    print(f"\n[导出成功] 所有模型运行数据已汇总至: {output_excel_path}")


if __name__ == "__main__":
    main()