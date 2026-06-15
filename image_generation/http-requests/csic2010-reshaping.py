# Imports originais
import math
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os 
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import gc 
from pyts.image import GramianAngularField, RecurrencePlot
import matplotlib.pyplot as plt
from tqdm import tqdm # Usando tqdm para a barra de progresso
from sequential import Sequential
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# --- Parte 1: Carregamento e Pré-processamento ---
print("Iniciando o carregamento e pré-processamento...")
def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

bad_requests = loadData('PreProcessedAnomalous.txt')
good_requests = loadData('PreprocessedNormalTraining.txt')
all_requests = bad_requests + good_requests

labels_Bad = [1] * len(bad_requests)
labels_Good = [0] * len(good_requests)
labels = labels_Bad + labels_Good
print ("Total requests : ",len(all_requests))
print ("Bad requests: ",len(bad_requests))
print ("Good requests: ",len(good_requests))

states = [0, 100, 1000]

image_reshapes = {
    "SEQ": Sequential(),
    "GASF": GramianAngularField(method="summation"),
    "GADF": GramianAngularField(method="difference"),
    "RPLOT": RecurrencePlot(threshold=None)
}

def generate_and_save_images(dataset_name, data, labels, base_dir, transformer, batch_size):
    print(f"  -> Gerando imagens para o conjunto '{dataset_name}'...")
    
    
    os.makedirs(os.path.join(base_dir, dataset_name, "benign"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, dataset_name, "malicious"), exist_ok=True)
    benign_count = 0
    malicious_count = 0
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"    Lotes {dataset_name}"):
        batch_data = data[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        
        generated_images = transformer.transform(batch_data)
        
        for j in range(len(generated_images)):
            image_to_save = generated_images[j]
            label = batch_labels[j]
            
            if label == 1:
                class_name = "malicious"
                file_path = os.path.join(base_dir, dataset_name, class_name, f"{class_name}_{malicious_count}.png")
                malicious_count += 1
            else:
                class_name = "benign"
                file_path = os.path.join(base_dir, dataset_name, class_name, f"{class_name}_{benign_count}.png")
                benign_count += 1
            
            colors = "gray" if (isinstance(transformer, RecurrencePlot) or isinstance(transformer, Sequential)) else "rainbow"
            plt.imsave(file_path, image_to_save, cmap=colors)
            
    gc.collect()
for state in states:
    print(f"\n{'='*20} INICIANDO EXECUÇÃO COM RANDOM_STATE = {state} {'='*20}")
    
    X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        all_requests, labels, test_size=0.3, random_state=state, stratify=labels
    )
    X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_text_temp, y_temp, test_size=0.5, random_state=state, stratify=y_temp
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, lowercase=False, max_features=1024)
    vectorizer.fit(X_text_train)
    X_train_text_vec = vectorizer.transform(X_text_train).toarray()
    X_val_text_vec = vectorizer.transform(X_text_val).toarray()
    X_test_text_vec = vectorizer.transform(X_text_test).toarray()
    
    X_train = X_train_text_vec
    X_val = X_val_text_vec
    X_test = X_test_text_vec
    
    k = 144 # Resultado: imagem 10x10
    selector = SelectKBest(f_classif, k=k)

    # Treina o 'selector' e já transforma o X_train
    X_train_kbest = selector.fit_transform(X_train, y_train)

    # APENAS transforma o X_val
    X_val_kbest = selector.transform(X_val)
    X_test_kbest = selector.transform(X_test)

    joblib.dump(selector, f"selector{state}csic.joblib")

    #joblib.dump(pca, f"pca{state}.joblib")
    joblib.dump(vectorizer, f"tfidf2{state}-csic.joblib")
    print(f"Dimensão final dos dados de treino (antes da imagem): {X_train.shape}")

    for image_type, transformer in image_reshapes.items():
        print(f"\n> Processando tipo de imagem: {image_type}")
        base_dir = f"../../images/CSIC-2010/{image_type}_state{state}" # Diretório específico para o state
        
        datasets_to_process = {
            "train": (X_train_kbest, y_train),
            "val": (X_val_kbest, y_val),
            "test": (X_test_kbest, y_test)
        }
        if isinstance(transformer, Sequential):
            transformer.fit(X_train_kbest)
            
        for name, (data, lbls) in datasets_to_process.items():
            generate_and_save_images(
                dataset_name=name,
                data=data,
                labels=lbls,
                base_dir=base_dir,
                transformer=transformer,
                batch_size=256 
            )
            
        print(f"Imagens {image_type} (state {state}) geradas e salvas com sucesso!")

print("\n--- PROCESSO TOTALMENTE CONCLUÍDO ---")