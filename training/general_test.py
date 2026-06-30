# coding=utf-8
import argparse
import logging
import os

import config
import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
from CNN import build_model
from MiniCNN import MiniCNN
from myutils import test, train
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import SwinForImageClassification, ViTForImageClassification

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "general_test.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("general_test")


def _resnet(variant: str, dropout: float) -> nn.Module:
    model = getattr(models, variant)(weights="DEFAULT")
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
    return model


def build_vision_model(model_name: str, dropout: float, model_path: str = None) -> nn.Module:
    registry = {
        "ResNet50": lambda: _resnet("resnet50", dropout),
        "ResNet18": lambda: _resnet("resnet18", dropout),
        "ConViT": lambda: timm.create_model(
            "convit_tiny", pretrained=True, num_classes=2
        ),
        "ConvNext-Nano": lambda: timm.create_model(
            "convnext_nano", pretrained=True, num_classes=2, in_chans=3
        ),
        "HybriDet": lambda: timm.create_model(
            "vit_tiny_r_s16_p8_224", pretrained=True, num_classes=2
        ),
        "FastViT": lambda: timm.create_model(
            "fastvit_t8", pretrained=True, num_classes=2
        ),
        "ViTB16": lambda: ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=2
        ),
        "Swin-Tiny": lambda: SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            num_labels=2,
            ignore_mismatched_sizes=True,
        ),
        "CustomCNN": lambda: build_model(
            arch={
                "name": "mid_3b_b",
                "blocks": [(32, 3, True), (64, 3, True), (128, 3, True)],
            },
            fc_dims=[512],
            dropout=0.5,
            use_bn=True,
        ),
        "MiniCNN": lambda: MiniCNN(num_classes=2),
        "DeiT-Tiny": lambda: timm.create_model(
            "deit_tiny_patch16_224", pretrained=True, num_classes=2
        ),
        "DeiT-Small": lambda: timm.create_model(
            "deit_small_patch16_224", pretrained=True, num_classes=2
        ),
        "EfficientViT-B0": lambda: timm.create_model(
            "efficientvit_b0", pretrained=True, num_classes=2
        ),
    }
    if model_name not in registry:
        logger.error("Modelo desconhecido: '%s'. Opções: %s", model_name, list(registry))
        raise ValueError(
            f"Modelo desconhecido: {model_name!r}. Opções: {list(registry)}"
        )

    logger.info("Construindo modelo '%s' (dropout=%.2f)", model_name, dropout)
    model = registry[model_name]()

    if model_path and os.path.exists(model_path):
        logger.info("Carregando checkpoint: %s", model_path)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    elif model_path:
        logger.warning("Checkpoint não encontrado: %s — iniciando do zero.", model_path)

    return model


def _unfreeze_head(model: nn.Module, model_name: str) -> None:
    """Garante que a cabeça de classificação sempre treina, mesmo com --tf."""
    if model_name in ("ResNet50", "ResNet18"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name in (
        "ConvNext-Nano",
        "ConViT",
        "HybriDet",
        "FastViT",
        "DeiT-Tiny",
        "DeiT-Small",
        "EfficientViT-B0",
    ):
        for p in model.head.parameters():
            p.requires_grad = True
    elif model_name in ("ViTB16", "Swin-Tiny"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    logger.debug("Cabeça de classificação descongelada para '%s'.", model_name)


def _get_transform(model_name: str) -> transforms.Compose:
    if model_name in ("MiniCNN", "CustomCNN"):
        return transforms.Compose(
            [
                transforms.Resize((config.RESOLUTION, config.RESOLUTION)),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_loaders(dirs: dict, batch_size: int, num_workers: int, transform) -> tuple:
    def _loader(split, shuffle):
        if not os.path.exists(dirs[split]):
            logger.error("Diretório do split '%s' não encontrado: %s", split, dirs[split])
            raise FileNotFoundError(f"Diretório não encontrado: {dirs[split]}")
        ds = datasets.ImageFolder(root=dirs[split], transform=transform)
        logger.info("Split '%s': %d amostras carregadas de '%s'.", split, len(ds), dirs[split])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return _loader("train", True), _loader("val", False), _loader("test", False)


def _append_csv(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    logger.info("Resultado salvo em '%s'.", path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf", action="store_true", help="Congelar camadas do backbone"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet50",
        choices=[
            "ResNet18",
            "ResNet50",
            "ConViT",
            "ConvNext-Nano",
            "ViTB16",
            "Swin-Tiny",
            "HybriDet",
            "FastViT",
            "MiniCNN",
            "CustomCNN",
            "DeiT-Tiny",
            "DeiT-Small",
            "EfficientViT-B0",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CSIC-2010", "FWAF", "Domain-Custom", "HTTP-PARAMS"],
    )
    parser.add_argument("--batch_size", type=int, help="Batch_size", default=16)
    parser.add_argument("--num_workers", type=int, help="workers", default=4)
    parser.add_argument(
        "--root",
        type=str,
        default="images/CSIC-2010",
        help="Diretório raiz das imagens",
    )

    return parser.parse_args()


def main(args):
    model_name = args.model
    num_epochs = args.epochs
    save_dir = f"saved_models/{model_name}/"
    results_csv = f"results/general_{model_name}.csv"

    logger.info("=" * 60)
    logger.info("Iniciando experimento | Modelo: %s | Dataset: %s | Épocas: %d",
                model_name, args.dataset, num_epochs)
    logger.info("Transfer learning (backbone congelado): %s", args.tf)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)
    cudnn.benchmark = True

    transform = _get_transform(model_name)

    for dataset_cfg in config.get_datasets(args.root):
        ds_name = dataset_cfg["name"]
        dirs = dataset_cfg["dirs"]

        logger.info("--- Processando dataset: %s ---", ds_name)

        try:
            train_loader, val_loader, test_loader = _make_loaders(
                dirs, args.batch_size, args.num_workers, transform
            )
        except FileNotFoundError as e:
            logger.error("Falha ao carregar dados para '%s': %s — pulando.", ds_name, e)
            continue

        logger.info("[%s] Dados carregados.", ds_name)

        save_model_name = f"{model_name}-{ds_name}-{config.RESOLUTION}-{args.dataset}"

        try:
            model = build_vision_model(model_name, config.P)
        except Exception as e:
            logger.exception("Falha ao construir modelo '%s': %s", model_name, e)
            continue

        for p in model.parameters():
            p.requires_grad = not args.tf
        _unfreeze_head(model, model_name)

        model = model.to(device)

        try:
            best_acc, best_epoch = train(
                model,
                num_epochs,
                train_loader,
                val_loader,
                output_dir=save_dir,
                model_name=save_model_name,
                device=device,
            )
        except Exception as e:
            logger.exception("Erro durante o treinamento de '%s' em '%s': %s",
                             model_name, ds_name, e)
            continue

        ckpt_path = os.path.join(save_dir, f"{save_model_name}_{num_epochs}.pth")
        if not os.path.exists(ckpt_path):
            logger.error("Checkpoint não encontrado após treino: %s", ckpt_path)
            continue

        logger.info("Carregando melhor checkpoint: %s", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        try:
            loss, acc, prec, rec, f1, auc = test(
                model, test_loader, model_name, device=device
            )
        except Exception as e:
            logger.exception("Erro durante o teste de '%s' em '%s': %s",
                             model_name, ds_name, e)
            continue

        logger.info("[%s] Melhor treino: acc=%.4f @ época %d", ds_name, best_acc, best_epoch)
        logger.info("[%s] Teste — Loss: %.4f | Acc: %.4f | Prec: %.4f | "
                    "Rec: %.4f | F1: %.4f | AUC: %.4f",
                    ds_name, loss, acc, prec, rec, f1, auc)

        _append_csv(
            results_csv,
            {
                "Image_Dataset": f"TEST-{ds_name}-{args.dataset}",
                "Model": model_name,
                "Epochs": num_epochs,
                "Test_Loss": loss,
                "Test_Acuracia": acc,
                "Test_Precisao": prec,
                "Test_Recall": rec,
                "Test_F1-Score": f1,
                "Test_ROC-AUC": auc,
                "Best_Acc_Train": best_acc,
                "Best_Epoch_Acc": best_epoch,
                "Num_Samples": config.NUM_SAMPLES,
                "Dropout": config.P,
                "Resolution": config.RESOLUTION,
                "Normalization": "Yes",
            },
        )

    logger.info("Experimento finalizado.")


if __name__ == "__main__":
    main(get_args())