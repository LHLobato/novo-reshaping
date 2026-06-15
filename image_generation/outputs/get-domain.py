from pathlib import Path
import pandas as pd

df_test = pd.read_csv("test.csv", index_col=False).drop(columns=['image'])
df_train = pd.read_csv("train.csv", index_col=False).drop(columns=['image'])
df_val = pd.read_csv("val.csv", index_col=False).drop(columns=['image'])
df_test.to_csv("test.csv", index=False)
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
