# coding=utf-8
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger("myutils")

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
    logger.info("Iniciando treino | modelo=%s | épocas=%d | AMP=%s | device=%s",
                model_name, num_epochs, use_amp, device)

    if not use_amp:
        logger.info("[%s] AMP desligado — rodando em FP32.", model_name)

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

    for epoch in tqdm(range(num_epochs), desc="Treinamento"):
        model.train()
        running_loss = 0.0
        all_preds, all_labels, all_proba = [], [], []
        nan_batches = 0

        for i, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Época {epoch + 1}/{num_epochs}")
        ):
            images, labels = images.to(device), labels.to(device)

            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = _get_logits(model(images))
                        loss = criterion(logits, labels) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits = _get_logits(model(images))
                    loss = criterion(logits, labels) / accumulation_steps
                    loss.backward()
            except Exception as e:
                logger.exception("Erro no forward/backward do batch %d (época %d): %s",
                                 i, epoch + 1, e)
                optimizer.zero_grad()
                continue

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
                nan_batches += 1
                logger.warning("NaN detectado no batch %d (época %d) — batch ignorado.",
                               i, epoch + 1)
                continue

            running_loss += loss.item() * accumulation_steps
            all_proba.extend(proba)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if nan_batches > 0:
            logger.warning("Época %d: %d batches ignorados por NaN.", epoch + 1, nan_batches)

        if not all_labels:
            logger.error("CRÍTICO: todos os batches falharam na época %d. Abortando treino.",
                         epoch + 1)
            break

        all_proba = np.nan_to_num(all_proba, nan=0.0)
        train_report = classification_report(all_labels, all_preds, output_dict=True)
        try:
            train_auc = roc_auc_score(all_labels, all_proba)
        except ValueError as e:
            logger.warning("Não foi possível calcular AUC no treino (época %d): %s",
                           epoch + 1, e)
            train_auc = 0.5

        # --- Validação ---
        model.eval()
        val_loss = 0.0
        va_preds, va_labels, va_proba = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                try:
                    ctx = torch.amp.autocast("cuda") if use_amp else torch.no_grad()
                    with ctx:
                        logits = _get_logits(model(images))
                        loss = criterion(logits, labels)
                except Exception as e:
                    logger.exception("Erro na validação (época %d): %s", epoch + 1, e)
                    continue

                val_loss += loss.item()
                proba = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                va_proba.extend(np.nan_to_num(proba, nan=0.0))
                va_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                va_labels.extend(labels.cpu().numpy())

        if not va_labels:
            logger.error("CRÍTICO: validação sem amostras válidas na época %d.", epoch + 1)
            continue

        val_report = classification_report(va_labels, va_preds, output_dict=True)
        val_accuracy = val_report["accuracy"]
        try:
            val_auc = roc_auc_score(va_labels, va_proba)
        except ValueError as e:
            logger.warning("Não foi possível calcular AUC na validação (época %d): %s",
                           epoch + 1, e)
            val_auc = 0.5

        if sched_mode == "epoch":
            scheduler.step(val_accuracy)

        logger.info(
            "Época %d/%d | Train Loss: %.4f | Train Acc: %.4f | Train AUC: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | Val AUC: %.4f",
            epoch + 1, num_epochs,
            running_loss / len(train_loader), train_report["accuracy"], train_auc,
            val_loss / len(val_loader), val_accuracy, val_auc,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_limit = 0
            ckpt = os.path.join(output_dir, f"{model_name}_{num_epochs}.pth")
            torch.save(model.state_dict(), ckpt)
            logger.info("Checkpoint salvo: %s (val_acc=%.4f)", ckpt, best_accuracy)
        else:
            patience_limit += 1
            logger.debug("Sem melhora. Patience: %d/%d.", patience_limit, patience)
            if patience_limit >= patience:
                logger.info("Early stopping ativado na época %d.", epoch + 1)
                break

    elapsed = (time.time() - start_time) / 60
    logger.info("Treinamento concluído em %.2f min. Melhor acc=%.4f @ época %d.",
                elapsed, best_accuracy, best_epoch)
    return best_accuracy, best_epoch


def test(model, test_loader, model_name, device="cuda"):
    use_amp = _use_amp(model_name)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    logger.info("Iniciando teste | modelo=%s | AMP=%s", model_name, use_amp)

    if not use_amp:
        logger.info("[%s] Teste em FP32.", model_name)

    test_loss = 0.0
    test_preds, test_labels, test_proba = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Teste")):
            images, labels = images.to(device), labels.to(device)

            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = _get_logits(model(images))
                        loss = criterion(logits, labels)
                else:
                    logits = _get_logits(model(images))
                    loss = criterion(logits, labels)
            except Exception as e:
                logger.exception("Erro no batch %d do teste: %s", i, e)
                continue

            test_loss += loss.item()
            proba = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            test_proba.extend(np.nan_to_num(proba, nan=0.0))
            test_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    if not test_labels:
        logger.error("DataLoader de teste vazio ou todos os batches falharam.")
        return 0, 0, 0, 0, 0, 0

    report = classification_report(test_labels, test_preds, output_dict=True)
    acc = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec = report["macro avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    try:
        auc = roc_auc_score(test_labels, test_proba)
    except ValueError as e:
        logger.warning("Não foi possível calcular AUC no teste: %s", e)
        auc = 0.5

    avg_loss = test_loss / len(test_loader)
    logger.info(
        "Resultado do teste | Loss: %.4f | Acc: %.4f | Prec: %.4f | "
        "Rec: %.4f | F1: %.4f | AUC: %.4f",
        avg_loss, acc, prec, rec, f1, auc,
    )
    return avg_loss, acc, prec, rec, f1, auc