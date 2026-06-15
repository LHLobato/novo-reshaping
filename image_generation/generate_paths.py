import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--typeimg", type=str, default="Gray-Scale",
                    choices=["Gray-Scale", "RGB Combined", "MKF-Combined", "CWT", "DeepInsight"])
args = parser.parse_args()

def simulate_on_disk(names, labels, path):
    records = []
    for i in range(len(names)):
        img_class = 'normal' if labels[i] == 0 else 'malicious'
        img_name  = f'{i}-{names[i]}.png'
        img_path  = os.path.join(path, img_class, img_name)
        records.append({
            "fname": img_name,
            "image_path": img_path,
            "label": labels[i],
            "domain": names[i]
        })
    return records

def simulate_filenames(df, path, batch_size=256):
    labels = df['malicious'].values
    names  = df['name'].values
    batches = range(0, len(df), batch_size)
    all_records = []

    for i in tqdm(batches, desc=f"Simulating [{path}]", unit="batch"):
        batch_names  = names[i: i+batch_size]
        batch_labels = labels[i: i+batch_size]
        records = simulate_on_disk(batch_names, batch_labels, path)
        all_records.extend(records)

    return pd.DataFrame(all_records)

# ── Load data ────────────────────────────────────────────────────────────────

train_df = pd.read_csv("outputs/train.csv", index_col=False)
val_df   = pd.read_csv("outputs/val.csv",   index_col=False)
test_df  = pd.read_csv("outputs/test.csv",  index_col=False)

# ── Simulate per image type (mirrors original logic) ─────────────────────────

if args.typeimg == "Gray-Scale":
    images = ["MKF", "GASF", "GADF", "RPLOT"]
    dfs    = [train_df, val_df, test_df]
    splits = ["train", "val", "test"]

    for img_type in images:
        for df, split in zip(dfs, splits):
            path = f"../images/{img_type}/{split}/"
            result = simulate_filenames(df, path)
            out_path = f"filenames-{img_type}-{split}.csv"
            result.to_csv(out_path, index=False)
            print(f"Saved {len(result)} records to {out_path}")

else:
    # single image type
    dfs    = [train_df, val_df, test_df]
    splits = ["train", "val", "test"]

    for df, split in zip(dfs, splits):
        path = f"../images/{args.typeimg}/{split}/"
        result = simulate_filenames(df, path)
        out_path = f"filenames-{args.typeimg}-{split}.csv"
        result.to_csv(out_path, index=False)
        print(f"Saved {len(result)} records to {out_path}")