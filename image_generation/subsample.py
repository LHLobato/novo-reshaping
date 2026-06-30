import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_SAMPLES = 100_000

df = pd.read_csv("dataset.csv")

print(f"Dataset original: {len(df)} amostras")
print(df["malicious"].value_counts(normalize=True))

# Subamostragem estratificada para 100k (mantém a proporção de classes)
df_sampled, _ = train_test_split(
    df,
    train_size=N_SAMPLES,
    stratify=df["malicious"],
    random_state=RANDOM_STATE,
)

print(f"\nApós subamostragem: {len(df_sampled)} amostras")
print(df_sampled["malicious"].value_counts(normalize=True))

# Split 70/15/15 estratificado (mesma lógica do pipeline original)
df_train, df_temp = train_test_split(
    df_sampled,
    test_size=0.30,
    stratify=df_sampled["malicious"],
    random_state=RANDOM_STATE,
)
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    stratify=df_temp["malicious"],
    random_state=RANDOM_STATE,
)

print(f"\nTrain: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
for nome, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"  {nome}: {d['malicious'].value_counts(normalize=True).to_dict()}")

df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

print("\nArquivos salvos: train.csv, val.csv, test.csv")