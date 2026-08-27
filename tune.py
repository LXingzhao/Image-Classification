import os
import sys
import time
import json
import csv
import yaml
import torch
import optuna
import argparse
import datetime
import train  # 导入你的 train.py 模块

# 全局变量：用于收集每次调参的具体参数、结果与耗时
trial_records = []

def save_records_to_files(records, save_dir):
    """将调参历史保存为 CSV 和 JSON 文件，防止乱码"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存为 CSV (使用 utf-8-sig 防止 Excel 打开乱码)
    csv_path = os.path.join(save_dir, "tuning_history.csv")
    fieldnames = ["trial_num", "lr", "batch_size", "weight_decay", "best_val_acc", "elapsed_sec", "status"]
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    # 2. 保存为 JSON
    json_path = os.path.join(save_dir, "tuning_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
        
    print(f"\n📂 调参历史已成功保存至:\n  - CSV:  {csv_path}\n  - JSON: {json_path}")

def objective(trial, base_args):
    """Optuna 目标函数：每次试验 (Trial) 生成一组超参数并运行训练"""
    start_time = time.time()
    
    # 1. 动态采样超参数
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    print(f"\n" + "=" * 60)
    print(f"🚀 [Optuna 调参轮次 {trial.number + 1}/{base_args.n_trials}] 启动训练")
    print(f"📌 当前超参数: LR={lr:.2e} | BatchSize={batch_size} | WeightDecay={weight_decay:.4f}")
    print(f"⏱️ 本轮早停阀值 (Patience): {base_args.patience_per_trial}")
    print("=" * 60)

    # 2. 模拟 sys.argv 参数传入 train.py（调参阶段强制 patience_per_trial = 5）
    sys.argv = [
        "train.py",
        "--model", base_args.model,
        "--dataset", base_args.dataset,
        "--sub_type", base_args.sub_type,
        "--patience", str(base_args.patience_per_trial)
    ]
    if base_args.config:
        sys.argv.extend(["--config", base_args.config])
    if base_args.dataset_config:
        sys.argv.extend(["--dataset_config", base_args.dataset_config])

    # 3. 动态解析配置文件路径
    dataset_input = base_args.dataset.replace("\\", "/")
    parts = dataset_input.split("/")
    dataset_base_name = parts[0]
    sub_type = parts[1] if len(parts) > 1 else base_args.sub_type

    dataset_config_path = base_args.dataset_config or os.path.join("configs", "datasets", f"{dataset_base_name}.yaml")
    
    if not base_args.config:
        model_filename = base_args.model if base_args.model.endswith(".yaml") else f"{base_args.model}.yaml"
        path_with_subtype = os.path.join("configs", "models", dataset_base_name, sub_type, model_filename) if sub_type else ""
        path_with_dataset = os.path.join("configs", "models", dataset_base_name, model_filename)
        fallback_path = os.path.join("configs", "models", model_filename)

        if path_with_subtype and os.path.exists(path_with_subtype):
            model_config_path = path_with_subtype
        elif os.path.exists(path_with_dataset):
            model_config_path = path_with_dataset
        else:
            model_config_path = fallback_path
    else:
        model_config_path = base_args.config

    # 4. 读取并更新配置中的超参数
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    if "train" not in model_cfg or model_cfg["train"] is None:
        model_cfg["train"] = {}

    model_cfg["train"]["lr"] = lr
    model_cfg["train"]["batch_size"] = batch_size
    model_cfg["train"]["weight_decay"] = weight_decay

    with open(model_config_path, "w", encoding="utf-8") as f:
        yaml.dump(model_cfg, f, allow_unicode=True)

    # 5. 读取合并后的完整配置
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    cfg["dataset"] = dataset_cfg.get("dataset", {})
    cfg["model"] = model_cfg.get("model", {})

    # 6. 执行训练并捕获异常
    best_val_acc = 0.0
    status = "SUCCESS"
    
    try:
        train.main()
    except torch.cuda.OutOfMemoryError:
        print(f"❌ [Trial {trial.number + 1}] 显存溢出 (OOM)，自动跳过该组合。")
        torch.cuda.empty_cache()
        status = "OOM"
    except Exception as e:
        print(f"❌ [Trial {trial.number + 1}] 运行异常: {e}")
        status = "ERROR"

    # 7. 从 history.json 读取训练结果
    if status == "SUCCESS":
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dataset_name = cfg['dataset']['name']
        model_type = cfg['model']['type']
        exp_name = f"{model_type}_exp1" if not model_type.endswith("_base") else f"{model_type[:-5]}_exp1"
        
        save_dir = os.path.join("outputs", dataset_name, sub_type, today_date, exp_name) if sub_type else os.path.join("outputs", dataset_name, today_date, exp_name)
        history_json_path = os.path.join(save_dir, "history.json")
        
        if os.path.exists(history_json_path):
            with open(history_json_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                best_val_acc = max(history.get("val_acc", [0.0]))

    elapsed_time = time.time() - start_time

    # 记录本轮调参日志
    trial_records.append({
        "trial_num": trial.number + 1,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "best_val_acc": best_val_acc,
        "elapsed_sec": round(elapsed_time, 2),
        "status": status
    })

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Optuna 超参数自动化搜索与训练")
    parser.add_argument("--model", type=str, default="vit_base")
    parser.add_argument("--dataset", type=str, default="SDNET2018")
    parser.add_argument("--sub_type", type=str, default="D")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--n_trials", type=int, default=20, help="搜索试验的总次数")
    parser.add_argument("--patience_per_trial", type=int, default=5, help="调参阶段单次 Trial 的早停轮数")
    parser.add_argument("--final_patience", type=int, default=10, help="使用最佳参数进行最终训练时的早停轮数")
    args = parser.parse_args()

    total_start_time = time.time()

    # Optuna 任务定义
    study = optuna.create_study(
        study_name="vit_hp_search",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("==================================================")
    print(f"📊 开始超参数自动化搜索 | 总计 Trial 轮次: {args.n_trials}")
    print(f"🎯 调参阶段单次早停 (Patience): {args.patience_per_trial} 轮")
    print(f"🎯 最终训练阶段早停 (Final Patience): {args.final_patience} 轮")
    print("==================================================")

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    best_params = study.best_params
    best_val_acc = study.best_value
    total_tuning_time = time.time() - total_start_time

    # ---------------------------------------------------------
    # 定位保存路径并将调参记录保存为表格文件
    # ---------------------------------------------------------
    dataset_input = args.dataset.replace("\\", "/")
    parts = dataset_input.split("/")
    dataset_base_name = parts[0]
    sub_type = parts[1] if len(parts) > 1 else args.sub_type

    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    tuning_log_dir = os.path.join("outputs", dataset_base_name, sub_type, today_date, "tuning_logs") if sub_type else os.path.join("outputs", dataset_base_name, today_date, "tuning_logs")
    
    save_records_to_files(trial_records, tuning_log_dir)

    # ---------------------------------------------------------
    # 写入最佳参数到 YAML
    # ---------------------------------------------------------
    if not args.config:
        model_filename = args.model if args.model.endswith(".yaml") else f"{args.model}.yaml"
        path_with_subtype = os.path.join("configs", "models", dataset_base_name, sub_type, model_filename) if sub_type else ""
        path_with_dataset = os.path.join("configs", "models", dataset_base_name, model_filename)
        fallback_path = os.path.join("configs", "models", model_filename)

        if path_with_subtype and os.path.exists(path_with_subtype):
            model_config_path = path_with_subtype
        elif os.path.exists(path_with_dataset):
            model_config_path = path_with_dataset
        else:
            model_config_path = fallback_path
    else:
        model_config_path = args.config

    print(f"\n[自动覆盖] 正在将最佳超参数写入配置文件: {model_config_path} ...")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    if "train" not in model_cfg or model_cfg["train"] is None:
        model_cfg["train"] = {}

    model_cfg["train"]["lr"] = float(best_params["lr"])
    model_cfg["train"]["batch_size"] = int(best_params["batch_size"])
    model_cfg["train"]["weight_decay"] = float(best_params["weight_decay"])

    with open(model_config_path, "w", encoding="utf-8") as f:
        yaml.dump(model_cfg, f, allow_unicode=True)
    print("✅ 配置文件写入完成！")

    # ---------------------------------------------------------
    # 启动最佳超参数的最终训练
    # ---------------------------------------------------------
    print("\n" + "=" * 30 + " 自动启动最佳模型最终训练 " + "=" * 30)
    print(f"🔥 使用最佳超参数: LR={best_params['lr']:.2e}, Batch={best_params['batch_size']}, WD={best_params['weight_decay']:.4f}")
    print(f"⏱️ 恢复最终早停阀值 (Patience): {args.final_patience} 轮")

    final_start_time = time.time()
    sys.argv = [
        "train.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--sub_type", args.sub_type,
        "--patience", str(args.final_patience)
    ]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.dataset_config:
        sys.argv.extend(["--dataset_config", args.dataset_config])

    train.main()
    final_train_time = time.time() - final_start_time

    # ---------------------------------------------------------
    # 整体流程统计总结
    # ---------------------------------------------------------
    print("\n" + "=" * 30 + " 全流程总结报告 " + "=" * 30)
    print(f"1. 调参总次数 (Trials): {args.n_trials} 轮")
    print(f"2. 调参阶段总耗时: {total_tuning_time / 60:.2f} 分钟 ({total_tuning_time:.1f} 秒)")
    print(f"3. 最佳验证集准确率 (Best Val Acc): {best_val_acc:.4f}")
    print(f"4. 最终模型训练耗时: {final_train_time / 60:.2f} 分钟 ({final_train_time:.1f} 秒)")
    print(f"5. 全流程总用时: {(total_tuning_time + final_train_time) / 60:.2f} 分钟")
    print("=" * 66)

if __name__ == "__main__":
    main()