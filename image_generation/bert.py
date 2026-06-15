import gc
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import urllib.parse

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,  help="Caminho para o modelo fine-tunado")
parser.add_argument("--max_len",    type=int, default=32,     help="Tamanho máximo de tokens")
parser.add_argument("--batch_size", type=int, default=64,     help="Batch size de inferência")
parser.add_argument("--n_layers",   type=int, default=3,      help="Número de camadas finais a empilhar como canais")
parser.add_argument("--dataset", type=str, required=True, choices=["CSIC-2010", "HTTP-PARAMS", "FWAF", "Domain"], help="Dataset")
parser.add_argument("--mode",       type=str, default="cls",
                    choices=["cls", "full", "multilayer"],
                    help="cls: só CLS token | full: sequência completa | multilayer: N camadas como canais RGB")

# ── Normalização ──────────────────────────────────────────────────────────────
def normalize_channel(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).astype(np.uint8)

def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        result = np.zeros_like(img, dtype=np.uint8)
        for c in range(3):
            result[:, :, c] = normalize_channel(img[:, :, c])
        return result
    return normalize_channel(img)

# ── Extração de Features em batch ─────────────────────────────────────────────
def extract_batch(domains: list) -> list:
    inputs = tokenizer(
        domains,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (Batch, Seq_Len, Hidden_Size)

    if args.mode == "cls":
        cls_features = hidden_states[-1][:, 0, :]          # (Batch, Hidden_Size)
        H = int(np.sqrt(hidden_size))
        W = hidden_size // H
        images = cls_features.view(-1, H, W).cpu().numpy() # (Batch, H, W)
        return [images[i] for i in range(images.shape[0])]

    elif args.mode == "full":
        last_layer = hidden_states[-1]                      # (Batch, Seq_Len, Hidden)
        reduced = last_layer.mean(dim=-1)                   # (Batch, Seq_Len)
        images  = reduced.cpu().numpy()                     # (Batch, Seq_Len)
        H = args.max_len
        W = 1
        return [images[i].reshape(H, W) for i in range(images.shape[0])]

    elif args.mode == "multilayer":
        assert args.n_layers <= len(hidden_states), \
        f"n_layers={args.n_layers} but model only has {len(hidden_states)} layers"

        def get_hw(h_size):
            h = int(np.sqrt(h_size))
            while h_size % h != 0:
                h -= 1
            w = h_size // h
            return h, w

        H, W = get_hw(hidden_size)

        selected = hidden_states[-args.n_layers:]
        channels = []
        for layer in selected:
            cls = layer[:, 0, :]           # (Batch, Hidden_Size)
            cls = cls.view(-1, H, W)       # (Batch, H, W)
            channels.append(cls)

        image_tensor = torch.stack(channels, dim=-1)  # (Batch, H, W, N_layers)
        images = image_tensor.cpu().numpy()
        return [images[i] for i in range(images.shape[0])]

# ── Salva batch em disco ──────────────────────────────────────────────────────
def on_disk(names, labels, images, path, offset):
    for i in range(len(images)):
        img_class = 'normal' if labels[i] == 0 else 'malicious'
        # Limpa o nome para evitar erro na criação de diretórios com /
        safe_name = str(names[i]).replace("/", "_").replace("\\", "_")[:50] 
        img_name  = f'{offset + i}-{safe_name}.png'
        img_path  = os.path.join(path, img_class, img_name)
        cv2.imwrite(img_path, normalize_image(images[i]))

# ── Pipeline principal ────────────────────────────────────────────────────────
def save_feature_maps(df: pd.DataFrame, path: str):
    os.makedirs(os.path.join(path, 'normal'),    exist_ok=True)
    os.makedirs(os.path.join(path, 'malicious'), exist_ok=True)

    names  = df['name'].values
    labels = df['malicious'].values
    bs     = args.batch_size

    for i in tqdm(range(0, len(names), bs),
                  desc=f"[{args.mode}] {path.split('/')[-2]}",
                  unit="batch"):
        batch_names  = names[i: i + bs].tolist()
        batch_labels = labels[i: i + bs]

        images = extract_batch(batch_names)
        on_disk(batch_names, batch_labels, images, path, offset=i)

        del images
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    return [d.strip() for d in data if len(d.strip()) > 0]

def loadFile(name):
    num_samples = 0
    filepath = os.path.join(os.getcwd(), name)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d)).strip()
        if len(d) > 0:
            result.append(d)
            num_samples += 1
            if num_samples >= 120000:
                break
    return result

# Helper para converter as listas dos datasets em DataFrames padronizados
def make_df(X, y):
    return pd.DataFrame({"name": X, "malicious": y})

if __name__ == "__main__":
    args = parser.parse_args()

    # ── Modelo e Configuração (Movido para cima para ficar global antes das execuções) ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Garantir que o tokenizer tem um pad_token (Crucial para modelos Llama/Qwen/GPT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        output_hidden_states=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"Model hidden size: {hidden_size} | Mode: {args.mode} | Layers: {args.n_layers}")

    # ── Datasets ────────────────────────────────────────────────────────────────
    if args.dataset == "CSIC-2010":
        bad_requests = loadData('../../datasets/CSIC-2010/PreProcessedAnomalous.txt')
        good_requests = loadData('../../datasets/CSIC-2010/PreprocessedNormalTraining.txt')
        
        all_requests = bad_requests + good_requests
        labels = [1] * len(bad_requests) + [0] * len(good_requests)

        X_train, X_temp, y_train, y_temp = train_test_split(all_requests, labels, test_size=0.3, random_state=0, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

        train_df = make_df(X_train, y_train)
        val_df   = make_df(X_val, y_val)
        test_df  = make_df(X_test, y_test)

    elif args.dataset == "FWAF":
        badQueries = loadFile('../../datasets/FWAF/badqueries.txt')
        validQueries = loadFile('../../datasets/FWAF/goodqueries.txt')

        allQueries = badQueries + validQueries
        labels = [1] * len(badQueries) + [0] * len(validQueries)

        X_train, X_temp, y_train, y_temp = train_test_split(allQueries, labels, test_size=0.3, random_state=0, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

        train_df = make_df(X_train, y_train)
        val_df   = make_df(X_val, y_val)
        test_df  = make_df(X_test, y_test)

    elif args.dataset == "HTTP-PARAMS":
        df = pd.read_csv("../../datasets/HTTPS-PARAMS/payload_full.csv", index_col=False)
        df.dropna(inplace=True)
        
        # Mapeia as strings para inteiros explicitamente
        df['label'] = df['label'].map({'norm': 0, 'anom': 1}).fillna(0).astype(int)

        payload = df['payload'].astype(str).values
        labels = df['label'].values

        X_train, X_temp, y_train, y_temp = train_test_split(payload, labels, test_size=0.3, random_state=0, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

        train_df = make_df(X_train, y_train)
        val_df   = make_df(X_val, y_val)
        test_df  = make_df(X_test, y_test)

    elif args.dataset == "Domain":
        train_df = pd.read_csv("outputs/train.csv", index_col=False)
        val_df   = pd.read_csv("outputs/val.csv",   index_col=False)
        test_df  = pd.read_csv("outputs/test.csv",  index_col=False)

        # Se os seus CSVs de domain usarem o nome 'label', renomeamos para 'malicious'
        for d in [train_df, val_df, test_df]:
            if 'label' in d.columns:
                d.rename(columns={'label': 'malicious'}, inplace=True)

    # ── O Dicionário Que Faltava ────────────────────────────────────────────────
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    print(f"Splits: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    folder_name = f"BERT-{args.mode}-{args.dataset}" if args.mode != "multilayer" else f"BERT-multilayer-{args.n_layers}ch-{args.dataset}"

    for split_name, split_df in splits.items():
        save_feature_maps(
            split_df,
            path=f"../images/{folder_name}/{split_name}/",
        )

    print("Concluído.")