import sys
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.builder import build_model_and_processor

def extract_features_for_subset(subset, model_type="swin_v2"):
    print(f"\n==================== 正在提取 [{subset}] 子集特征 (模型: {model_type}) ====================")
    
    data_dir = f"data/SDNET2018/{subset}"
    if not Path(data_dir).exists():
        print(f"⚠️ 警告: 数据集路径不存在: {data_dir}，跳过。")
        return None, None, None

    # 1. 智能模糊匹配配置文件
    subset_config_dir = Path(f"configs/models/SDNET2018/{subset}")
    config_path = None
    main_keyword = model_type.lower().split('-')[0].split('_')[0] 

    if subset_config_dir.exists():
        for y in subset_config_dir.glob("*.yaml"):
            if main_keyword in y.name.lower() or model_type.lower() in y.name.lower():
                config_path = y
                break
                
    if not config_path or not config_path.exists():
        general_dir = Path(r"configs/models/SDNET2018/D_P_W")
        if general_dir.exists():
            for y in general_dir.glob("*.yaml"):
                if main_keyword in y.name.lower() or model_type.lower() in y.name.lower():
                    config_path = y
                    break
                    
    if not config_path or not config_path.exists():
        all_yamls = list(Path("configs/models").glob("**/*.yaml"))
        for y in all_yamls:
            if main_keyword in y.name.lower():
                config_path = y
                break

    if not config_path or not config_path.exists():
        print(f"⚠️ 提示: 未能自动找到与 '{model_type}' 匹配的 yaml 文件，将临时借用 swin_v2_base.yaml 配置。")
        config_path = Path(f"configs/models/SDNET2018/{subset}/swin_v2_base.yaml")
        
    print(f"[{subset}子集] 最终使用配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_name = config['model']['name']
    img_size = config.get('data', {}).get('img_size', 256)

    # 2. 兼容两种输出目录结构（同时检查 outputs/SDNET2018_{sub} 和 outputs/SDNET2018/{sub}）
    search_dirs = [
        Path(f"outputs/SDNET2018_{subset}"),
        Path(f"outputs/SDNET2018/{subset}")
    ]
    for other_sub in ["W", "D", "P"]:
        if other_sub != subset:
            search_dirs.append(Path(f"outputs/SDNET2018_{other_sub}"))
            search_dirs.append(Path(f"outputs/SDNET2018/{other_sub}"))

    checkpoint_candidates = []
    for search_dir in search_dirs:
        if search_dir.exists():
            candidates = [
                p for p in search_dir.glob("**/checkpoints/best.pth") 
                if main_keyword in p.parents[1].name.lower() or model_type.lower() in p.parents[1].name.lower()
            ]
            if not candidates:
                candidates = list(search_dir.glob("**/checkpoints/best.pth"))
                
            if candidates:
                checkpoint_candidates = candidates
                print(f"[{subset}子集] 在目录 {search_dir} 中成功命中权重！")
                break
    
    if not checkpoint_candidates:
        raise FileNotFoundError(f"⚠️ 在所有输出目录下均未找到任何 best.pth 权重文件！")
    
    checkpoint_path = sorted(checkpoint_candidates, key=lambda p: p.parents[1].name)[-1]
    print(f"[{subset}子集] 锁定最新权重: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 准备数据
    temp_dataset = ImageFolder(root=data_dir)
    class_names_original = temp_dataset.classes  
    
    print(f"[{subset}子集] 数据集路径: {data_dir}")
    print(f"[{subset}子集] 类别标签: {class_names_original}")

    model, _ = build_model_and_processor(model_name, len(temp_dataset.classes), temp_dataset.classes)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"[{subset}子集] 样本总数: {len(dataset)} 张图片")

    # 4. 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"[{subset}子集] 模型权重加载成功！")

    # 5. Hook 提取特征
    features_list = []
    labels_list = []
    
    activation = {}
    def hook_fn(module, input, output):
        feat = input[0] if isinstance(input, tuple) and len(input) > 0 else output
        if feat.dim() > 2:
            feat = feat.mean(dim=list(range(1, feat.dim() - 1)))
        elif feat.dim() == 1:
            feat = feat.unsqueeze(0)
        activation['feature'] = feat.detach().cpu()

    target_module = None
    for name, module in model.named_modules():
        if name in ['classifier', 'head', 'fc'] or name.endswith('.classifier') or name.endswith('.head'):
            target_module = module
            break
    if target_module is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Linear):
                target_module = module
                break

    if target_module is None:
        raise AttributeError(f"[{subset}] 未能在模型中找到合适的分类头或线性层来绑定 Hook。")

    hook_handle = target_module.register_forward_hook(hook_fn)

    print(f"[{subset}子集] 开始前向传播提取特征...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)  
            features_list.append(activation['feature'].numpy())
            labels_list.append(labels.numpy())

    hook_handle.remove()

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    mapped_labels = np.array([class_names_original[l] for l in labels])
    
    print(f"[{subset}子集] 特征提取完毕，特征矩阵形状: {features.shape}")
    return features, mapped_labels, subset

def get_custom_palette(unique_labels):
    fixed_colors = {
        'CD': '#1E88E5',  # 亮宝蓝 (D类有裂缝)
        'CP': '#00ACC1',  # 青蓝色 (P类有裂缝)
        'CW': '#3949AB',  # 深靛蓝 (W类有裂缝)
        'UD': '#E53935',  # 鲜艳红 (D类无裂缝)
        'UP': '#FB8C00',  # 亮橙色 (P类无裂缝)
        'UW': '#D81B60'   # 玫瑰红 (W类无裂缝)
    }
    
    palette = {}
    for label in unique_labels:
        if label in fixed_colors:
            palette[label] = fixed_colors[label]
        else:
            palette[label] = '#1E88E5' if 'C' in label.upper() else '#E53935'
    return palette

def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "swin_v2"
    print(f"🚀 当前选择的模型类型: [{model_type}]")

    subsets = ["W", "D", "P"]
    all_features = []
    all_labels = []
    
    global_all_labels = ['CD', 'CP', 'CW', 'UD', 'UP', 'UW']
    global_palette = get_custom_palette(global_all_labels)

    # ==================== 第一步：分别生成各自的单子集图 ====================
    for subset in subsets:
        res = extract_features_for_subset(subset, model_type)
        if res[0] is None:
            continue
        features, mapped_labels, _ = res
        
        all_features.append(features)
        all_labels.append(mapped_labels)

        print(f"正在对 [{subset}] 子集进行 t-SNE 降维...")
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        X_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=mapped_labels,
            hue_order=sorted(np.unique(mapped_labels)),
            palette=global_palette,  
            alpha=0.8,
            s=5,
            edgecolor='none'
        )
        plt.title(f't-SNE Feature Visualization - SDNET2018_{subset} ({model_type})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes')
        plt.tight_layout()
        
        output_name = f'tsne_result_{subset}_{model_type}.png'
        plt.savefig(output_name, dpi=300)
        plt.close()
        print(f"✅ 单子集图已成功保存: {output_name}\n")

    # ==================== 第二步：生成合在一起的综合图 ====================
    if all_features:
        print(f"\n==================== 正在合并 W, D, P 数据生成综合 t-SNE 图 ({model_type}) ====================")
        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"合并后总特征矩阵形状: {combined_features.shape}")
        print(f"包含的所有类别: {np.unique(combined_labels)}")
        
        print("正在进行大合集 t-SNE 降维...")
        tsne_combined = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        X_tsne_combined = tsne_combined.fit_transform(combined_features)

        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            x=X_tsne_combined[:, 0],
            y=X_tsne_combined[:, 1],
            hue=combined_labels,
            hue_order=sorted(np.unique(combined_labels)),
            palette=global_palette,  
            alpha=0.8,
            s=2,  
            edgecolor='none'
        )
        plt.title(f't-SNE Feature Visualization - SDNET2018 Combined (W, D, P) ({model_type})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='All Classes')
        plt.tight_layout()
        
        combined_output_name = f'tsne_result_combined_{model_type}.png'
        plt.savefig(combined_output_name, dpi=300)
        plt.close()
        print(f"✅ 综合合集图已成功保存为: {combined_output_name}！")

if __name__ == '__main__':
    main()