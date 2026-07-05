import argparse, os, sys
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unified_model import UnifiedMultimodalModel
from train import MultimodalNPZDataset, multimodal_collate, classification_loss, focal_loss, cox_partial_log_likelihood

def objective(trial):
    # Hyperparameters to search
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    cls_weight = trial.suggest_float("cls_weight", 0.5, 2.0)
    surv_weight = trial.suggest_float("surv_weight", 0.5, 2.0)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    gamma = trial.suggest_float("gamma", 1.5, 3.0)
    alpha = trial.suggest_float("alpha", 0.3, 0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_ds = MultimodalNPZDataset("/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data/train.npz")
    val_ds = MultimodalNPZDataset("/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data/val.npz")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate, num_workers=0)
    
    modality_input_dims = [arr.shape[1] for arr in train_ds.modalities]
    ae_latent_dims = [64, 256, 256, 256, 128][:len(modality_input_dims)]
    
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=ae_latent_dims,
        transformer_cfg={"d_model": d_model, "nhead": nhead, "num_layers": num_layers, "dim_feedforward": d_model*2, "dropout": dropout, "use_cls_token": True},
        num_classes=6
    ).to(device)
    
    model.setup_pathway_attention("/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data/soft_module_mask.npz")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = torch.tensor([1.0, 1.5, 2.9, 3.8, 3.4, 4.6]).to(device)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            modalities, labels, durations, events = batch
            modalities = [m.to(device) for m in modalities]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, log_risk, attn_weights, fused, reconstructions, pathway_weights = model(
                modalities, modality_mask=None, add_noise_to_ae=True, return_reconstructions=True
            )
            
            loss_cls = focal_loss(logits, labels, alpha=class_weights*alpha, gamma=gamma)
            loss_surv = cox_partial_log_likelihood(log_risk, durations.to(device), events.to(device))
            loss_recon = sum(torch.nn.functional.mse_loss(rec, mod.detach()) for rec, mod in zip(reconstructions, modalities))
            
            loss = cls_weight * loss_cls + surv_weight * loss_surv + 0.3 * loss_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                modalities, labels, durations, events = batch
                modalities = [m.to(device) for m in modalities]
                labels = labels.to(device)
                
                logits, log_risk, attn_weights, fused, reconstructions, pathway_weights = model(
                    modalities, modality_mask=None, add_noise_to_ae=False, return_reconstructions=True
                )
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if patience_counter >= 5:
            break
    
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--output", type=str, default="/tmp/optuna_results.csv")
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    
    # Save results
    df = study.trials_dataframe()
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
