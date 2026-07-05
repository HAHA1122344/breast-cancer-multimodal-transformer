import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models.unified_model import UnifiedMultimodalModel
from train import MultimodalNPZDataset, multimodal_collate, compute_classification_metrics
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, data_path, output_dir="results", num_classes=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = MultimodalNPZDataset(data_path)
    # Binarize mutation modality if present
    if len(dataset.modalities) > 4:
        dataset.modalities[4] = (dataset.modalities[4] > 0).astype(np.float32)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=multimodal_collate)

    # 获取模态维度
    modality_input_dims = [arr.shape[1] for arr in dataset.modalities]
    num_modalities = len(modality_input_dims)

    # Use explicit latent dims matching the checkpoint [64, 256, 256, 256, 128]
    ae_latent_dims = [64, 256, 256, 256, 128][:num_modalities]
    if len(ae_latent_dims) < num_modalities:
        ae_latent_dims.extend([128] * (num_modalities - len(ae_latent_dims)))

    # 加载模型
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=ae_latent_dims,
        transformer_cfg={"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512, "dropout": 0.3, "use_cls_token": True},
        num_classes=num_classes
    ).to(device)

    # Initialize new modules
    if hasattr(model, "setup_cross_modal_attention"):
        model.setup_cross_modal_attention(n_pathways=20, n_modalities=num_modalities)
    if hasattr(model, "setup_modality_gating"):
        model.setup_modality_gating(n_modalities=num_modalities, embed_dim=256)

    # Initialize pathway attention if checkpoint contains it
    state_dict = torch.load(model_path, map_location="cpu")
    if any(k.startswith("pathway_attention.") for k in state_dict.keys()):
        # Determine mask type from checkpoint shape
        soft_key = "pathway_attention.soft_membership"
        use_soft = soft_key in state_dict
        import os
        soft_path = "../data/soft_module_mask.npz"
        hard_path = "../data/pathway_mask.npz"
        mask_path = soft_path if use_soft and os.path.exists(soft_path) else hard_path
        model.setup_pathway_attention(mask_path, use_soft_membership=use_soft)
    # Initialize new modules if checkpoint expects them
    if hasattr(model, "setup_cross_modal_attention"):
        model.setup_cross_modal_attention(n_pathways=20, n_modalities=num_modalities)
    if hasattr(model, "setup_modality_gating"):
        model.setup_modality_gating(n_modalities=num_modalities, embed_dim=256)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 收集预测结果
    all_labels = []
    all_preds = []
    all_probs = []
    all_risks = []
    all_durations = []
    all_events = []
    all_attn_weights = []
    
    with torch.no_grad():
        for batch in loader:
            modalities, labels, durations, events = batch
            modalities = [m.to(device) for m in modalities]
            
            outputs = model(modalities, modality_mask=None, add_noise_to_ae=False)
            if len(outputs) == 5:
                logits, log_risk, attn_weights, fused, _ = outputs
            else:
                logits, log_risk, attn_weights, fused = outputs
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            risks = torch.exp(log_risk)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_risks.extend(risks.cpu().numpy().flatten())
            all_durations.extend(durations.cpu().numpy())
            all_events.extend(events.cpu().numpy())
            all_attn_weights.extend(attn_weights.cpu().numpy())
    
    # 计算指标
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    
    # 分类报告
    class_names = ['LumA', 'LumB', 'Her2', 'claudin-low', 'Basal', 'Normal'][:num_classes]
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # C-Index
    c_index = concordance_index(all_durations, -np.array(all_risks), all_events)
    print(f"\nC-Index (生存分析): {c_index:.4f}")
    
    # 模态注意力分析
    attn_array = np.array(all_attn_weights)
    print(f"\n模态注意力权重 (平均):")
    mod_names = ['临床', 'mRNA', 'CNA', '甲基化', '突变'][:num_modalities]
    for i in range(num_modalities):
        print(f"  模态{i} ({mod_names[i]}): {attn_array[:, i].mean():.4f}")
    
    # 可视化
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150)
    print(f"\n混淆矩阵图已保存: {output_dir}/confusion_matrix.png")
    
    # 注意力分布图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([attn_array[:, i] for i in range(min(3, num_modalities))])
    plt.xticks(range(1, min(4, num_modalities+1)), mod_names[:min(3, num_modalities)])
    plt.ylabel('注意力权重')
    plt.title('模态注意力分布')

    plt.subplot(1, 2, 2)
    plt.bar(mod_names[:num_modalities], attn_array.mean(axis=0))
    plt.ylabel('平均注意力权重')
    plt.title('平均模态重要性')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_weights.png', dpi=150)
    print(f"注意力分布图已保存: {output_dir}/attention_weights.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/val.npz")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_classes", type=int, default=6)
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path, args.output_dir, args.num_classes)
