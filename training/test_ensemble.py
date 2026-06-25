# ensemble_experiment.py  — versão corrigida com get_args

import os
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ensemble import FeatureExtractor, registry


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv",  type=str, required=True,
                        help="Caminho do CSV de resultados (ex: results/ensemble.csv)")
    parser.add_argument("--model_dir",    type=str, required=True,
                        help="Diretório dos checkpoints (ex: saved_models)")
    parser.add_argument("--model_name",   type=str, required=True,
                        choices=["Swin-Tiny", "ViTB16", "DeiT-Small",
                                 "ResNet50", "ResNet18", "ConViT",
                                 "ConvNext-Nano", "HybriDet", "FastViT",
                                 "MiniCNN", "CustomCNN", "DeiT-Tiny", "EfficientViT-B0"])
    parser.add_argument("--type_img",     type=str, required=True,
                        help="Tipo de imagem (ex: GASF, GADF, RPLOT, SEQ)")
    parser.add_argument("--dataset",      type=str, required=True,
                        help="Nome do dataset (ex: CSIC-2010)")
    parser.add_argument("--image_dir",    type=str, required=True,
                        help="Diretório raiz das imagens (ex: images/CSIC-2010)")
    parser.add_argument("--features_dir", type=str, default="features",
                        help="Diretório para salvar/carregar features")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--dropout",      type=float, default=0.0)
    return parser.parse_args()


def _append_csv(path: str, row: dict):
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _already_evaluated(results_csv: str, model_name: str, dataset_name: str, clf_name: str) -> bool:
    if not os.path.exists(results_csv):
        return False
    df = pd.read_csv(results_csv)
    return (
        (df["Model"]   == model_name)  &
        (df["Dataset"] == dataset_name) &
        (df["Clf"]     == clf_name)
    ).any()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(
        args.model_dir,
        args.model_name,
        f"{args.model_name}-{args.type_img}-{config.RESOLUTION}-{args.dataset}_{config.NUM_EPOCHS}.pth",
    )

    dataset_cfgs = config.get_datasets(args.image_dir)
    dataset_cfg = next((d for d in dataset_cfgs if d["name"] == args.type_img), None)
    if dataset_cfg is None:
        raise ValueError(f"type_img {args.type_img!r} não encontrado em config.get_datasets.")

    print(f"\n{'='*60}")
    print(f"Modelo: {args.model_name} | Dataset: {args.dataset} | Tipo: {args.type_img}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    feat_val_path  = os.path.join(args.features_dir, f"val-{args.model_name}-{args.dataset}-{args.type_img}.npz")
    feat_test_path = os.path.join(args.features_dir, f"test-{args.model_name}-{args.dataset}-{args.type_img}.npz")

    if not (os.path.exists(feat_val_path) and os.path.exists(feat_test_path)):
        extractor = FeatureExtractor(
            device=device,
            model_name=args.model_name,
            model_path=ckpt_path,
            dropout=args.dropout,
        )
        extractor.set_loaders(
            dirs=dataset_cfg["dirs"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        extractor.extract_and_save(args.features_dir, args.dataset, args.type_img)
        del extractor
        torch.cuda.empty_cache()
    else:
        print(f"[SKIP] Features já existem.")


    (X_val, y_val), (X_test, y_test) = FeatureExtractor.load_features(
        args.features_dir, args.model_name, args.dataset, args.type_img
    )
    print(f"Features — val: {X_val.shape} | test: {X_test.shape}")

    for clf_name, clf_factory in registry.items():
        if _already_evaluated(args.results_csv, args.model_name, args.dataset, clf_name):
            print(f"[SKIP] {clf_name} já avaliado.")
            continue

        print(f"  → {clf_name}...", end=" ", flush=True)
        try:
            clf = clf_factory()
            clf.fit(X_val, y_val)
            y_pred = clf.predict(X_test)

            try:
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_prob = clf.decision_function(X_test)
                else:
                    y_prob = None
            except Exception:
                y_prob = None

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

            print(f"acc={acc:.4f}" + (f" | auc={auc:.4f}" if auc else ""))

            _append_csv(args.results_csv, {
                "Model":     args.model_name,
                "Dataset":   args.dataset,
                "Type":      args.type_img,
                "Clf":       clf_name,
                "Accuracy":  acc,
                "Precision": prec,
                "Recall":    rec,
                "F1":        f1,
                "ROC_AUC":   auc,
            })

        except Exception as e:
            print(f"ERRO: {e}")
            _append_csv(args.results_csv, {
                "Model": args.model_name, "Dataset": args.dataset,
                "Type": args.type_img, "Clf": clf_name,
                "Accuracy": None, "Precision": None,
                "Recall": None, "F1": None, "ROC_AUC": None,
            })


if __name__ == "__main__":
    main(get_args())