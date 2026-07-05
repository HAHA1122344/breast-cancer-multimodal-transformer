# Breast Cancer Multimodal Transformer

A unified multimodal transformer for breast cancer molecular subtype classification and survival prediction, incorporating pathway-guided attention mechanisms.

## Features

- **Multimodal Fusion**: Clinical (80-dim), mRNA (500 genes), CNA (500 genes), Methylation (500 genes), Mutations (173 genes)
- **Pathway-Guided Attention**: Soft module membership based on gene co-expression networks (20 modules)
- **Dual Task Learning**: Simultaneous classification (PAM50/Claudin-low subtypes) and survival prediction (Cox proportional hazards)
- **Training Tricks**: Focal Loss, mixup augmentation, AdamW optimizer, cosine annealing LR

## Model Performance (v7, CPU training)

| Metric | Value |
|--------|-------|
| Test Accuracy | 63% |
| Test Macro F1 | 58% |
| Test C-Index | 0.591 |
| LumA Recall | 70% |

## Dataset

- **METABRIC** (Nature 2012, Nat Commun 2016)
- 2,509 breast cancer patients
- 5 modalities: clinical, mRNA, CNA, methylation, mutations
- 6 molecular subtypes: LumA, LumB, Her2, claudin-low, Basal, Normal

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py \
    --train_npz data/train.npz \
    --val_npz data/val.npz \
    --output_dir checkpoints \
    --epochs 100 \
    --pathway_mask_path data/soft_module_mask.npz
```

## Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/test.npz \
    --output_dir results \
    --num_classes 6
```

## Visualization

```bash
python visualize.py \
    --model_path checkpoints/best_model.pt \
    --data_npz data/test.npz \
    --out_dir figures \
    --num_classes 6
```

## Paper

See `paper_cn_v2.pdf` for the full paper draft.

## Data Availability

Preprocessed data files are available upon request.
Original data: [cBioPortal METABRIC](https://www.cbioportal.org/study/brca_metabric).

## Citation

```bibtex
@article{liu2025breast,
  title={A Unified Multimodal Transformer with Pathway-Guided Attention for Breast Cancer Molecular Subtype Classification and Survival Prediction},
  author={Liu, Shuxing},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License