# coding=utf-8
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

_TRANSFORMER_MODELS = frozenset(
    {
        "ViTB16",
        "Swin-Tiny",
        "DeiT-Tiny",
        "DeiT-Small",
        "ConvNext-Nano",
        "MaxViT-Tiny",
    }
)

_FP32_MODELS = frozenset(
    {
        "CoAtNet",
    }
)
_HF_MODELS = frozenset(
    {
        "ViTB16",
        "Swin-Tiny",
    }
)

_HEAD_MODELS = frozenset(
    {
        "ConvNext-Nano",
        "ConViT",
        "HybriDet",
        "FastViT",
        "DeiT-Tiny",
        "DeiT-Small",
        "MaxViT-Tiny",
        "EfficientViT-B0",
    }
)


def _use_amp(model_name: str) -> bool:
    return model_name not in _FP32_MODELS


def _use_transformer_schedule(model_name: str) -> bool:
    return model_name in _TRANSFORMER_MODELS


def _get_logits(outputs):
    """Extrai logits independente de ser saída HF ou tensor puro."""
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _make_optimizer(model, model_name: str):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if model_name in _FP32_MODELS:
        return torch.optim.AdamW(params, lr=5e-5, weight_decay=0.05)
    if model_name in _TRANSFORMER_MODELS:
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.05)
    return torch.optim.AdamW(params, lr=5e-4, weight_decay=0.005)


def _make_scheduler(
    optimizer, model_name, num_epochs, steps_per_epoch, accumulation_steps
):
    if _use_transformer_schedule(model_name) or model_name in _FP32_MODELS:
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_epochs * steps_per_epoch)
            // accumulation_steps,
            num_training_steps=(num_epochs * steps_per_epoch) // accumulation_steps,
        ), "step"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.1
    ), "epoch"


def train(
    model,
    num_epochs,
    train_loader,
    val_loader,
    output_dir,
    model_name,
    device="cuda",
    accumulation_steps=1,
):
    os.makedirs(output_dir, exist_ok=True)

    use_amp = _use_amp(model_name)
    if not use_amp:
        print(f"[{model_name}] AMP desligado — rodando em FP32.")

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, model_name)
    scheduler, sched_mode = _make_scheduler(
        optimizer, model_name, num_epochs, len(train_loader), accumulation_steps
    )

    best_accuracy = float("-inf")
    best_epoch = 0
    patience = 5
    patience_limit = 0
    start_time = time.time()

    optimizer.zero_grad()
    print(f"[{model_name}] Iniciando treinamento — AMP={use_amp}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels, all_proba = [], [], []

        for i, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Época {epoch + 1}/{num_epochs}")
        ):
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = _get_logits(model(images))
                    loss = criterion(logits, labels) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = _get_logits(model(images))
                loss = criterion(logits, labels) / accumulation_steps
                loss.backward()

            is_last_batch = (i + 1) == len(train_loader)
            if (i + 1) % accumulation_steps == 0 or is_last_batch:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                if sched_mode == "step":
                    scheduler.step()

            proba = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            if np.isnan(proba).any():
                print(f"  [Aviso] Batch {i}: NaN detectado, batch ignorado.")
                continue

            running_loss += loss.item() * accumulation_steps
            all_proba.extend(proba)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if not all_labels:
            print("CRÍTICO: todos os batches falharam. Abortando.")
            break

        all_proba = np.nan_to_num(all_proba, nan=0.0)
        train_report = classification_report(all_labels, all_preds, output_dict=True)
        try:
            train_auc = roc_auc_score(all_labels, all_proba)
        except ValueError:
            train_auc = 0.5

        model.eval()
        val_loss = 0.0
        va_preds, va_labels, va_proba = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                ctx = torch.amp.autocast("cuda") if use_amp else torch.no_grad()
                with ctx:
                    logits = _get_logits(model(images))
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                proba = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                va_proba.extend(np.nan_to_num(proba, nan=0.0))
                va_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                va_labels.extend(labels.cpu().numpy())

        val_report = classification_report(va_labels, va_preds, output_dict=True)
        val_accuracy = val_report["accuracy"]
        try:
            val_auc = roc_auc_score(va_labels, va_proba)
        except ValueError:
            val_auc = 0.5

        if sched_mode == "epoch":
            scheduler.step(val_accuracy)

        print(
            f"Época {epoch + 1}/{num_epochs} | "
            f"Train Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_report['accuracy']:.4f} | Train AUC: {train_auc:.4f}\n"
            f"              | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_limit = 0
            ckpt = os.path.join(output_dir, f"{model_name}_{num_epochs}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Checkpoint salvo: {ckpt}")
        else:
            patience_limit += 1
            if patience_limit >= patience:
                print("Early stopping.")
                break

    print(f"Treinamento concluído em {(time.time() - start_time) / 60:.2f} min.")
    return best_accuracy, best_epoch


def test(model, test_loader, model_name, device="cuda"):
    use_amp = _use_amp(model_name)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    if not use_amp:
        print(f"[{model_name}] Teste em FP32.")

    test_loss = 0.0
    test_preds, test_labels, test_proba = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Teste"):
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = _get_logits(model(images))
                    loss = criterion(logits, labels)
            else:
                logits = _get_logits(model(images))
                loss = criterion(logits, labels)

            test_loss += loss.item()
            proba = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            test_proba.extend(np.nan_to_num(proba, nan=0.0))
            test_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    if not test_labels:
        print("Erro: DataLoader de teste vazio.")
        return 0, 0, 0, 0, 0, 0

    report = classification_report(test_labels, test_preds, output_dict=True)
    acc = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec = report["macro avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    try:
        auc = roc_auc_score(test_labels, test_proba)
    except ValueError:
        auc = 0.5

    avg_loss = test_loss / len(test_loader)
    print(
        f"Test Loss: {avg_loss:.4f} | Acc: {acc:.4f} | "
        f"Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
    )
    return avg_loss, acc, prec, rec, f1, auc
