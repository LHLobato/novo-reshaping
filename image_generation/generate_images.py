import gc

import pandas as pd
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
import numpy as np
import os 
import cv2 
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser
from tqdm import tqdm 
import warnings
import numpy as np
from ssqueezepy import cwt
import matplotlib.pyplot as plt
from pyDeepInsight import ImageTransformer

warnings.filterwarnings('ignore', category=UserWarning, module='pyts')

parser = ArgumentParser()

parser.add_argument("--typeimg", type=str, default="Gray-Scale", choices=["Gray-Scale", "RGB Combined", "MKF-Combined", "CWT", "DeepInsight"], help="Tipo imagem")
args = parser.parse_args()

def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:  
        result = np.zeros_like(img, dtype=np.uint8)
        for c in range(3):
            channel = img[:, :, c]
            min_val, max_val = channel.min(), channel.max()
            if max_val == min_val:
                print(f"[WARN] canal {c} constante (min=max={min_val:.4f})")
                result[:, :, c] = 0
            else:
                result[:, :, c] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return result
    else:  
        min_val, max_val = img.min(), img.max()
        if max_val == min_val:
            print(f"[WARN] imagem constante (min=max={min_val:.4f})")
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def get_cwt(arr, n_scales=42):
    cwt_final = []
    for fv in arr:
        Wx, _ = cwt(fv, wavelet='morlet')  # mínimo de escalas
        scalogram = np.abs(Wx)
        # pega exatamente n_scales escalas por interpolação
        indices = np.linspace(0, scalogram.shape[0]-1, n_scales).astype(int)
        scalogram = scalogram[indices, :]  # (42, 42)
        cwt_final.append(scalogram)
    return cwt_final

def on_disk(names, labels, images, path):
    for i in range(len(images)):
        img_class = 'normal' if labels[i] == 0 else 'malicious'
        img_name  = f'{i}-{names[i]}.png'
        img_path  = os.path.join(path, img_class, img_name)

        if args.typeimg == "CWT":
            import matplotlib.cm as cm
            img = normalize_image(images[i])
            colored = (cm.viridis(img / 255.0)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(img_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        else:
            img = normalize_image(images[i])
            cv2.imwrite(img_path, img)

def save_images_on_disk(transformer, scaler,  df:pd.DataFrame, path:str, batch_size=256):
    os.makedirs(os.path.join(path, 'normal'),    exist_ok=True)
    os.makedirs(os.path.join(path, 'malicious'), exist_ok=True)
    X = scaler.transform(df.drop(columns=['name', 'malicious']).values)
    labels = df['malicious'].values 
    names = df['name'].values 
    batches = range(0, len(X), batch_size)

    if args.typeimg in ("RGB Combined", "MKF-Combined"):
        for i in tqdm(batches, desc=f"RGB [{path.split('/')[-2]}]", unit="batch"):
            batch_to_process = X[i: i+batch_size]
            r_gasf = transformer[0].transform(batch_to_process)
            g_gadf = transformer[1].transform(batch_to_process)
            b_rplot = transformer[2].transform(batch_to_process)

            n, h, w = r_gasf.shape
            images = np.zeros((n, h, w, 3), dtype=np.float32)
            images[:, :, :, 0] = r_gasf   
            images[:, :, :, 1] = g_gadf   
            images[:, :, :, 2] = b_rplot  

            on_disk(names[i: i+batch_size], labels[i: i+batch_size], images, path)
            del images, r_gasf, b_rplot, g_gadf, batch_to_process
            gc.collect()
    elif args.typeimg == "CWT":

        for i in tqdm(batches, desc=f"{args.typeimg} [{path.split('/')[-2]}]", unit="batch"):
            batch_to_process = X[i: i+batch_size]
            images = get_cwt(batch_to_process)
            on_disk(names[i: i+batch_size], labels[i: i+batch_size], images, path)

            del images, batch_to_process
            gc.collect()
    elif args.typeimg == "DeepInsight":
        for i in tqdm(batches, desc=f"DeepInsight [{path.split('/')[-2]}]", unit="batch"):
            batch_to_process = X[i: i+batch_size]
            images = transformer.transform(batch_to_process)
            on_disk(names[i: i+batch_size], labels[i: i+batch_size], images, path)
            del images, batch_to_process
            gc.collect()



    else:
        for i in tqdm(batches, desc=f"{args.typeimg} [{path.split('/')[-2]}]", unit="batch"):
            batch_to_process = X[i: i+batch_size]
            images = transformer.transform(batch_to_process)
            on_disk(names[i: i+batch_size], labels[i: i+batch_size], images, path)

            del images, batch_to_process
            gc.collect()





train_460k = pd.read_csv("outputs/train.csv", index_col=False)
val_460k   = pd.read_csv("outputs/val.csv",   index_col=False)
test_460k  = pd.read_csv("outputs/test.csv",  index_col=False)

print(len(train_460k), len(val_460k), len(test_460k))

y_train = train_460k['malicious']
y_val   = val_460k['malicious']
y_test  = test_460k['malicious']

transformers = [MarkovTransitionField(), GramianAngularField(), GramianAngularField(method="difference"), RecurrencePlot(threshold=None)]
dfs = [train_460k, val_460k, test_460k]
paths = ["train/", "val/", "test/"]
scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(train_460k.drop(columns=['name', 'malicious']).values)

if args.typeimg == "Gray-Scale":
    images = ["MKF", "GASF", "GADF", "RPLOT"]
    for j in range(len(images)):
        for i in range(len(paths)):
            save_images_on_disk(transformers[j], scaler,  dfs[i], f"../images/{images[j]}/{paths[i]}")

elif args.typeimg == "RGB Combined":

    images = ["RGB Combined"]
    for j in range(len(images)):
        for i in range(len(paths)):
            save_images_on_disk(transformers, scaler,  dfs[i], f"../images/{images[j]}/{paths[i]}")
elif args.typeimg == "CWT":
    for i in range(len(paths)):
            save_images_on_disk(None, scaler, dfs[i], f"../images/CWT/{paths[i]}")

elif args.typeimg == "DeepInsight":
    print("Mapeando a topologia das features com DeepInsight (t-SNE)... Isso pode levar alguns instantes.")
    di_transformer = ImageTransformer(feature_extractor='tsne', pixels=7)
    

    X_train_scaled = scaler.transform(train_460k.drop(columns=['name', 'malicious']).values)
    di_transformer.fit(X_train_scaled)
    
    for i in range(len(paths)):
        save_images_on_disk(di_transformer, scaler, dfs[i], f"../images/DeepInsight/{paths[i]}")            
else:
    transformers = [GramianAngularField(), GramianAngularField(method="difference"), MarkovTransitionField()]
    images = ["MKF-Combined"]
    for j in range(len(images)):
        for i in range(len(paths)):
            save_images_on_disk(transformers, scaler,  dfs[i], f"../images/{images[j]}/{paths[i]}")
