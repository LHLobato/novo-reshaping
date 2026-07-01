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
domains_df = pd.read_csv("dataset.csv")

domain_names = domains_df.iloc[:, 0].values
numerical_features = domains_df.drop(columns=['name', 'malicious']).values
labels = domains_df['malicious'].values

print(f"Carregado: {len(labels)} amostras")
print(f"Features de texto (domínios): {domain_names.shape}")
print(f"Features numéricas: {numerical_features.shape}")

N_SUBSAMPLE = 100_000  # Tamanho total desejado para as imagens
N_TRAIN = int(N_SUBSAMPLE * 0.70) # 70.000
N_VAL = int(N_SUBSAMPLE * 0.15)   # 15.000
N_TEST = int(N_SUBSAMPLE * 0.15)  # 15.000

states = [0]

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

    # 1. Split inicial no dataset completo (70% Treino, 15% Val, 15% Teste)
    X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        domain_names, labels, test_size=0.3, random_state=state, stratify=labels
    )
    X_num_train, X_num_temp, _, _ = train_test_split(
        numerical_features, labels, test_size=0.3, random_state=state, stratify=labels
    )
    X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_text_temp, y_temp, test_size=0.5, random_state=state, stratify=y_temp
    )
    X_num_val, X_num_test, _, _ = train_test_split(
        X_num_temp, y_temp, test_size=0.5, random_state=state, stratify=y_temp
    )
    
    # 2. Transformações (Fit apenas no treino)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, lowercase=False, max_features=1024)
    X_train_text_vec = vectorizer.fit_transform(X_text_train).toarray()
    X_val_text_vec = vectorizer.transform(X_text_val).toarray()
    X_test_text_vec = vectorizer.transform(X_text_test).toarray()
    
    scaler = MinMaxScaler()
    X_train_num_scaled = scaler.fit_transform(X_num_train)
    X_val_num_scaled = scaler.transform(X_num_val)
    X_test_num_scaled = scaler.transform(X_num_test)
    
    X_train_full = np.hstack((X_train_text_vec, X_train_num_scaled))
    X_val_full = np.hstack((X_val_text_vec, X_val_num_scaled))
    X_test_full = np.hstack((X_test_text_vec, X_test_num_scaled))
    
    k = 144 
    selector = SelectKBest(f_classif, k=k)

    X_train_kbest = selector.fit_transform(X_train_full, y_train)
    X_val_kbest = selector.transform(X_val_full)
    X_test_kbest = selector.transform(X_test_full)

    # 3. Subamostragem PÓS-transformação com as proporções exatas
    # Treino
    if len(y_train) > N_TRAIN:
        X_train_img, _, y_train_img, _ = train_test_split(
            X_train_kbest, y_train, train_size=N_TRAIN, stratify=y_train, random_state=state
        )
    else:
        X_train_img, y_train_img = X_train_kbest, y_train    

    # Validação
    if len(y_val) > N_VAL:
        X_val_img, _, y_val_img, _ = train_test_split(
            X_val_kbest, y_val, train_size=N_VAL, stratify=y_val, random_state=state
        )
    else:
        X_val_img, y_val_img = X_val_kbest, y_val 

    # Teste
    if len(y_test) > N_TEST:
        X_test_img, _, y_test_img, _ = train_test_split(
            X_test_kbest, y_test, train_size=N_TEST, stratify=y_test, random_state=state
        )
    else:
        X_test_img, y_test_img = X_test_kbest, y_test 

    joblib.dump(selector, f"selector{state}-gray.joblib")
    joblib.dump(vectorizer, f"tfidf2{state}-gray.joblib")
    print(f"Dimensão final dos dados de treino (antes da imagem): {X_train_img.shape}")

    # 4. Geração de Imagens
    for image_type, transformer in image_reshapes.items():
        print(f"\n> Processando tipo de imagem: {image_type}")
        base_dir = f"../datasets/TF-IDF-KBest-img/gray/{image_type}_state{state}"
        
        # Agora o dicionário usa as variáveis corretamente subamostradas
        datasets_to_process = {
            "train": (X_train_img, y_train_img),
            "val": (X_val_img, y_val_img),
            "test": (X_test_img, y_test_img)
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