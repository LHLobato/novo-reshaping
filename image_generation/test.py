import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

 # grade refinada + ponto de ancoragem em 100%
train_sizes = np.concatenate([np.linspace(0.10, 0.25, 16), [1.0]])
scorer = make_scorer(f1_score)

meu_modelo = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",       
        min_samples_leaf=2,
        n_jobs=3,
        random_state=42,
    )

df = pd.read_csv("dataset.csv",index_col=False)

X = df.drop(columns=["name", "malicious"]).values 
y = df["malicious"].values 

cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
sizes_s, _, val_s = learning_curve(
    meu_modelo, X, y,
    train_sizes=train_sizes,
    cv=cv_strat,
    scoring=scorer,
    n_jobs=3,
    shuffle=True,
    random_state=42,
)

# Aleatória (KFold comum, sem estratificar)
cv_rand = KFold(n_splits=3, shuffle=True, random_state=42)
sizes_r, _, val_r = learning_curve(meu_modelo, X, y, train_sizes=train_sizes,
                                     cv=cv_rand, scoring=scorer, n_jobs=3)

mean_s, std_s = val_s.mean(axis=1), val_s.std(axis=1)
mean_r, std_r = val_r.mean(axis=1), val_r.std(axis=1)

asymptote = mean_s[-1]  # ou média das duas em 100%
target = 0.99 * asymptote

idx_s = np.argmax(mean_s >= target)
idx_r = np.argmax(mean_r >= target)

print(f"Estratificada: {sizes_s[idx_s]:.0f} amostras, F1={mean_s[idx_s]:.4f}")
print(f"Aleatória:     {sizes_r[idx_r]:.0f} amostras, F1={mean_r[idx_r]:.4f}")

plt.errorbar(sizes_s, mean_s, yerr=std_s, label='Estratificada', marker='o')
plt.errorbar(sizes_r, mean_r, yerr=std_r, label='Aleatória', marker='s')
plt.axhline(target, ls='--', color='gray', label='99% da assíntota')
plt.xlabel('Tamanho da amostra'); plt.ylabel('F1'); plt.legend()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight') 