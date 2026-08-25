# tune.py
import os
import sys
import yaml
import optuna
import argparse
import train  # 导入你的 train.py 模块

def objective(trial, base_args):
    """
    Optuna 目标函数：每次试验 (Trial) 生成一组超参数并运行训练
    """
    # 1. 动态采样超参数 (针对 RTX 5090 优化)
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # 利用 5090 大显存
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    # 2. 模拟 sys.argv 参数传入 train.py 的 argparse
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

    # 3. 动态解析本次 Trial 的配置文件路径
    # 这段逻辑与 train.py 解析路径保持一致
    args = train.argparse.ArgumentParser().parse_known_args()[0]
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

    # 4. 临时读取并重写配置字典中的超参数
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "train" not in cfg:
        cfg["train"] = {}

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}
        cfg["dataset"] = dataset_cfg.get("dataset", {})
        if "train" in dataset_cfg and dataset_cfg["train"]:
            cfg["train"].update(dataset_cfg["train"])

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
        cfg["model"] = model_cfg.get("model", {})
        if "train" in model_cfg and model_cfg["train"]:
            cfg["train"].update(model_cfg["train"])

    # **注入 Optuna 生成的超参数**
    cfg["train"]["lr"] = lr
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["weight_decay"] = weight_decay

    # 5. 覆盖写回临时模型配置文件，保证 train.py 读取到最新参数
    with open(model_config_path, "w", encoding="utf-8") as f:
        yaml.dump(model_cfg, f)

    # 6. 执行训练
    print(f"\n[Optuna Trial {trial.number}] 尝试参数: LR={lr:.2e}, BatchSize={batch_size}, WeightDecay={weight_decay:.4f}")
    
    # 捕获异常，防止某次 CUDA OOM 导致整个搜索过程挂掉
    try:
        train.main()
    except torch.cuda.OutOfMemoryError:
        print(f"[Optuna Trial {trial.number}] 显存溢出 (OOM)，跳过该组参数。")
        torch.cuda.empty_cache()
        return 0.0

    # 7. 从全局日志中获取本次试验的验证集最佳结果
    # 你的 train.py 中存储了全局最佳 validation accuracy
    # 为了传回给 Optuna，重新读取生成的 history.json 获取最高 val_acc
    today_date = train.datetime.datetime.now().strftime("%Y-%m-%d")
    dataset_name = cfg['dataset']['name']
    model_type = cfg['model']['type']
    exp_name = f"{model_type}_exp1" if not model_type.endswith("_base") else f"{model_type[:-5]}_exp1"
    
    if sub_type:
        save_dir = os.path.join("outputs", dataset_name, sub_type, today_date, exp_name)
    else:
        save_dir = os.path.join("outputs", dataset_name, today_date, exp_name)

    history_json_path = os.path.join(save_dir, "history.json")
    
    if os.path.exists(history_json_path):
        with open(history_json_path, "r", encoding="utf-8") as f:
            history = train.json.load(f)
            best_val_acc = max(history["val_acc"])
            return best_val_acc
    else:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Optuna 超参数自动化搜索")
    parser.add_argument("--model", type=str, default="dinov2_base")
    parser.add_argument("--dataset", type=str, default="SDNET2018")
    parser.add_argument("--sub_type", type=str, default="D")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--n_trials", type=int, default=15, help="搜索试验的总次数")
    parser.add_argument("--patience_per_trial", type=int, default=5, help="单次 Trial 的早停轮数(设小一点可加速调参)")
    args = parser.parse_args()

    # 创建 Optuna 学习任务，目标是最大化 Validation Accuracy
    study = optuna.create_study(
        study_name="clip_hp_search",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)  # 使用 TPE 贝叶斯优化算法
    )

    print("================ 开始超参数自动化搜索 ================")
    print(f"搜索次数: {args.n_trials} 次 | 目标指标: 验证集准确率 (val_acc)")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n" + "="*20 + " 调参结果总结 " + "="*20)
    print(f"最佳验证集准确率 (Best Val Acc): {study.best_value:.4f}")
    print("最佳超参数组合 (Best Params):")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.6f}")
        else:
            print(f"  - {key}: {value}")
    print("="*54)

if __name__ == "__main__":
    main()