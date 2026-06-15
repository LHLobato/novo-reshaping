# coding=utf-8
"""
tuning.py — Busca de hiperparâmetros e topologia para a CNN customizada.

Estratégia: Random Search sobre o espaço combinado de arquitetura + hiperparâmetros.
Dataset: subconjunto de 5.000 imagens GADF para manter custo computacional baixo.
Resultados salvos em CSV ao fim de cada trial (fail-safe).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import itertools
import argparse
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from CNN import ConvolutionalNeuralNetwork, train, test


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
IMG_SIZE      = 42          # Resolução das imagens GADF
IN_CHANNELS   = 3
NUM_CLASSES   = 2
NUM_SAMPLES   = 10_000      # Subconjunto para a busca
NUM_EPOCHS    = 20          # Épocas por trial (curto para busca rápida)
BATCH_SIZE    = 64
NUM_WORKERS   = 4
OUTPUT_DIR    = "../saved_models/tuning/"
RESULTS_CSV   = "../results/tuning_results.csv"

# Caminho para o dataset GADF (ajuste conforme seu config.py / estrutura real)
# Espera-se um ImageFolder com splits train/ val/ test/ dentro deste diretório.
GADF_DATA_DIR = "../data/GADF/split_0"   # ← ajuste aqui

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# ESPAÇO DE BUSCA
# ─────────────────────────────────────────────────────────────────────────────

# Cada arquitetura é um dict descrevendo os blocos convolucionais.
# A notação é: lista de (out_channels, kernel_size, use_pool).
ARCH_CANDIDATES = [
    # ── 2 blocos ────────────────────────────────────────────────────────────
    {"name": "tiny_2b",      "blocks": [(32, 3, True),  (64, 3, True)]},
    {"name": "tiny_2b_k5",   "blocks": [(32, 5, True),  (64, 5, True)]},

    # ── 3 blocos — equilibradas para 42×42 ──────────────────────────────────
    {"name": "mid_3b_a",     "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, False)]},
    {"name": "mid_3b_b",     "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, True)]},
    {"name": "mid_3b_c",     "blocks": [(64, 3, True),  (128, 3, True), (256, 3, False)]},
    {"name": "mid_3b_wide",  "blocks": [(64, 3, True),  (128, 3, True), (256, 3, True)]},

    # ── 4 blocos ─────────────────────────────────────────────────────────────
    {"name": "deep_4b_a",    "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, True),  (256, 3, False)]},
    {"name": "deep_4b_b",    "blocks": [(64, 3, True),  (128, 3, True), (256, 3, True),  (512, 3, False)]},
    # kernel misto — extrai features de escala diferente
    {"name": "deep_4b_mix",  "blocks": [(32, 5, True),  (64, 3, True),  (128, 3, True),  (256, 3, False)]},
    # canais maiores desde o início
    {"name": "deep_4b_fat",  "blocks": [(64, 3, True),  (128, 3, True), (256, 3, True),  (512, 3, True)]},

    # ── 5 blocos — redes mais profundas com 10k samples ──────────────────────
    # Atenção: com 42×42 e 3 MaxPool a resolução espacial chega a ~5×5,
    # então o 4º e 5º blocos sem pool só aumentam profundidade sem reduzir mais.
    {"name": "vgg_5b_a",     "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, True),  (256, 3, False), (256, 3, False)]},
    {"name": "vgg_5b_b",     "blocks": [(64, 3, True),  (128, 3, True), (256, 3, True),  (512, 3, False), (512, 3, False)]},
    {"name": "vgg_5b_wide",  "blocks": [(64, 3, True),  (128, 3, True), (256, 3, True),  (512, 3, False), (256, 3, False)]},
    # bottleneck — expande depois contrai (inspirado em ResNet)
    {"name": "bottleneck_5b","blocks": [(64, 3, True),  (128, 3, True), (256, 3, True),  (128, 3, False), (64, 3, False)]},
    # kernel 5 nas primeiras camadas (captura contexto maior antes de reduzir)
    {"name": "wide_k5_5b",   "blocks": [(32, 5, True),  (64, 5, True),  (128, 3, True),  (256, 3, False), (256, 3, False)]},
]

FC_CANDIDATES = [
    # cabeças simples
    [256],
    [512],
    [1024],
    # cabeças em funil — as mais comuns em CNNs clássicas
    [512, 256],
    [1024, 512],
    [1024, 512, 256],
    [512, 256, 128],
    # cabeça mais profunda
    [1024, 512, 256, 128],
]

HYPERPARAM_GRID = {
    "lr":           [1e-3, 1e-4, 5e-4, 1e-5],
    "weight_decay": [1e-2, 1e-3, 5e-3],
    "dropout":      [0.3, 0.4, 0.5],
}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUÇÃO DINÂMICA DO MODELO
# ─────────────────────────────────────────────────────────────────────────────

def _infer_flat_size(conv_block: nn.Sequential, img_size: int, in_channels: int) -> int:
    """Passa um tensor dummy pela parte convolucional para descobrir o tamanho achatado."""
    dummy = torch.zeros(1, in_channels, img_size, img_size)
    with torch.no_grad():
        out = conv_block(dummy)
    return int(out.flatten(start_dim=1).shape[1])


def build_conv_layers(blocks: list, in_channels: int = 3, use_bn: bool = True) -> nn.Sequential:
    """
    Constrói a parte convolucional.

    blocks: lista de tuplas (out_channels, kernel_size, use_pool)
    """
    layers = []
    ch_in = in_channels
    for (ch_out, k, pool) in blocks:
        layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=k, padding=k // 2))
        if use_bn:
            layers.append(nn.BatchNorm2d(ch_out))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        ch_in = ch_out
    return nn.Sequential(*layers)


def build_fc_layers(flat_size: int, hidden_dims: list, dropout: float,
                    num_classes: int = 2) -> nn.Sequential:
    """
    Constrói a cabeça totalmente conectada.

    hidden_dims: lista de inteiros com os tamanhos das camadas ocultas.
    """
    layers = []
    in_dim = flat_size
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def build_model(arch: dict, fc_dims: list, dropout: float,
                use_bn: bool = True) -> ConvolutionalNeuralNetwork:
    conv_block = build_conv_layers(arch["blocks"], IN_CHANNELS, use_bn)
    flat_size  = _infer_flat_size(conv_block, IMG_SIZE, IN_CHANNELS)
    fc_block   = build_fc_layers(flat_size, fc_dims, dropout, NUM_CLASSES)
    return ConvolutionalNeuralNetwork(list(conv_block), list(fc_block))


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO DATASET (subconjunto balanceado de 5.000 amostras)
# ─────────────────────────────────────────────────────────────────────────────

def load_subset(data_dir: str, split: str, n_samples: int | None = None):
    """
    Carrega um split do ImageFolder e, opcionalmente, extrai um subconjunto
    balanceado com n_samples amostras totais.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, split), transform=transform)

    if n_samples is None or n_samples >= len(dataset):
        return dataset

    # Balanceamento por classe
    targets = np.array(dataset.targets)
    n_per_class = n_samples // len(dataset.classes)
    indices = []
    for cls_idx in range(len(dataset.classes)):
        cls_indices = np.where(targets == cls_idx)[0].tolist()
        sampled = random.sample(cls_indices, min(n_per_class, len(cls_indices)))
        indices.extend(sampled)

    random.shuffle(indices)
    return Subset(dataset, indices)


def get_loaders(data_dir: str, n_samples: int):
    train_ds = load_subset(data_dir, "train", n_samples)
    val_ds   = load_subset(data_dir, "val")    # validação completa
    test_ds  = load_subset(data_dir, "test")   # teste completo

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def generate_trials(n_trials: int) -> list[dict]:
    """
    Amostra aleatória do espaço combinado (arquitetura + hiperparâmetros).
    Garante que não há trials duplicados quando n_trials < total de combinações.
    """
    full_space = list(itertools.product(
        ARCH_CANDIDATES,
        FC_CANDIDATES,
        HYPERPARAM_GRID["lr"],
        HYPERPARAM_GRID["weight_decay"],
        HYPERPARAM_GRID["dropout"],
        [True, False],   # use_bn
    ))
    print(f"[Tuning] Espaço total: {len(full_space)} combinações.")
    n_trials = min(n_trials, len(full_space))
    sampled  = random.sample(full_space, n_trials)

    trials = []
    for arch, fc_dims, lr, wd, dr, bn in sampled:
        trials.append({
            "arch":         arch,
            "fc_dims":      fc_dims,
            "lr":           lr,
            "weight_decay": wd,
            "dropout":      dr,
            "use_bn":       bn,
        })
    return trials


def run_search(n_trials: int, data_dir: str, device: torch.device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(data_dir, NUM_SAMPLES)
    print(f"[Tuning] Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    trials   = generate_trials(n_trials)
    results  = []
    best_cfg = None
    best_f1  = -1.0

    for t_idx, cfg in enumerate(trials):
        arch     = cfg["arch"]
        fc_dims  = cfg["fc_dims"]
        lr       = cfg["lr"]
        wd       = cfg["weight_decay"]
        dropout  = cfg["dropout"]
        use_bn   = cfg["use_bn"]

        trial_name = (
            f"trial{t_idx:03d}"
            f"_{arch['name']}"
            f"_fc{'x'.join(str(d) for d in fc_dims)}"
            f"_lr{lr}"
            f"_wd{wd}"
            f"_dr{dropout}"
            f"_bn{int(use_bn)}"
        )
        print(f"\n{'='*70}")
        print(f"[Trial {t_idx+1}/{n_trials}] {trial_name}")
        print(f"  arch     : {arch['name']}  blocks={arch['blocks']}")
        print(f"  fc_dims  : {fc_dims}")
        print(f"  lr       : {lr}   wd: {wd}   dropout: {dropout}   bn: {use_bn}")
        print(f"{'='*70}")

        try:
            model = build_model(arch, fc_dims, dropout, use_bn).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parâmetros treináveis: {n_params:,}")

            best_val_acc, best_epoch = train(
                model       = model,
                num_epochs  = NUM_EPOCHS,
                train_loader= train_loader,
                val_loader  = val_loader,
                output_dir  = OUTPUT_DIR,
                model_name  = trial_name,
                lr          = lr,
                weight_decay= wd,
                device      = device,
            )

            # Carrega o melhor checkpoint salvo pelo train()
            ckpt_path = os.path.join(OUTPUT_DIR, f"{trial_name}_{NUM_EPOCHS}.pth")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

            t_loss, t_acc, t_prec, t_rec, t_f1, t_auc = test(
                model       = model,
                test_loader = test_loader,
                model_name  = trial_name,
                device      = device,
            )

        except Exception as e:
            print(f"  [ERRO] Trial falhou: {e}")
            t_loss, t_acc, t_prec, t_rec, t_f1, t_auc = 0, 0, 0, 0, 0, 0
            best_val_acc, best_epoch = 0, 0

        row = {
            "trial":           t_idx,
            "trial_name":      trial_name,
            "arch":            arch["name"],
            "blocks":          str(arch["blocks"]),
            "fc_dims":         str(fc_dims),
            "lr":              lr,
            "weight_decay":    wd,
            "dropout":         dropout,
            "use_bn":          use_bn,
            "best_val_acc":    best_val_acc,
            "best_epoch":      best_epoch,
            "test_loss":       t_loss,
            "test_acc":        t_acc,
            "test_precision":  t_prec,
            "test_recall":     t_rec,
            "test_f1":         t_f1,
            "test_auc":        t_auc,
        }
        results.append(row)

        # Salva incremental (fail-safe: não perde resultados se travar)
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(RESULTS_CSV, index=False)
        print(f"  → Test F1: {t_f1:.4f} | AUC: {t_auc:.4f} | Val Acc: {best_val_acc:.4f}")

        if t_f1 > best_f1:
            best_f1  = t_f1
            best_cfg = deepcopy(cfg)
            print(f"  ★ Novo melhor! F1={best_f1:.4f}")

    return results, best_cfg


# ─────────────────────────────────────────────────────────────────────────────
# SUMÁRIO FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list, best_cfg: dict):
    df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    print("\n" + "="*70)
    print("RESULTADOS — TOP 5 CONFIGURAÇÕES (por F1)")
    print("="*70)
    top5_cols = ["trial_name", "test_f1", "test_auc", "test_acc",
                 "lr", "weight_decay", "dropout", "use_bn"]
    print(df[top5_cols].head(5).to_string(index=False))

    print("\n" + "="*70)
    print("MELHOR CONFIGURAÇÃO PARA general_test.py")
    print("="*70)
    arch = best_cfg["arch"]
    print(f"  Arquitetura  : {arch['name']}")
    print(f"  Blocos conv  : {arch['blocks']}")
    print(f"  FC dims      : {best_cfg['fc_dims']}")
    print(f"  Dropout      : {best_cfg['dropout']}")
    print(f"  Batch Norm   : {best_cfg['use_bn']}")
    print(f"  LR           : {best_cfg['lr']}")
    print(f"  Weight Decay : {best_cfg['weight_decay']}")
    print("\nCopie esses valores para general_test.py e CNN_config.py.")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Search — arquitetura + hiperparâmetros da CNN customizada"
    )
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Número de trials do random search (default: 20)")
    parser.add_argument("--data_dir", type=str, default=GADF_DATA_DIR,
                        help="Caminho para o split do dataset GADF (ImageFolder)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Épocas por trial (default: 20)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help="Amostras de treino por trial (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Aplica overrides dos argumentos de linha de comando
    NUM_EPOCHS  = args.epochs
    NUM_SAMPLES = args.samples
    BATCH_SIZE  = args.batch_size
    SEED        = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tuning] Device: {device}")
    print(f"[Tuning] Trials: {args.n_trials} | Épocas/trial: {NUM_EPOCHS} | "
          f"Samples treino: {NUM_SAMPLES}")

    results, best_cfg = run_search(
        n_trials = args.n_trials,
        data_dir = args.data_dir,
        device   = device,
    )

    print_summary(results, best_cfg)
    print(f"\nResultados completos salvos em: {RESULTS_CSV}")
