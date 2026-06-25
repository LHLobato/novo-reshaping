from general_test import build_vision_model, _get_transform, _make_loaders
import torch 
from torchvision import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 
from sklearn.svm import LinearSVC 
from sklearn.ensemble import VotingClassifier, StackingClassifier
from itertools import combinations
from tqdm import tqdm
from myutils import _use_amp
import numpy as np 
from torch.utils.data import DataLoader
import os 

def _all_estimators(base: dict) -> list[tuple[str, object]]:
    return [(name, clf()) for name, clf in base.items()]

BASE = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",       
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ),
    "KNN": lambda: KNeighborsClassifier(
        n_neighbors=11,
        metric="cosine",           
        algorithm="brute",         
        n_jobs=-1,
    ),
    "XGB": lambda: XGBClassifier(
        n_estimators=400,
        max_depth=4,               
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.4,      
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
    "LinearSVC": lambda: LinearSVC(
        C=0.1,                    
        max_iter=2000,
        random_state=42,
    ),
}

_voting_entries = {
    f"Voting_{'_'.join(names)}": (
        lambda names=names: VotingClassifier(
            estimators=[(n, clf()) for n, clf in BASE.items() if n in names],
            voting="hard",
            n_jobs=-1,
        )
    )
    for r in range(2, len(BASE) + 1)
    for names in combinations(BASE.keys(), r)
}

_stacking_entries = {
    f"Stacking_{'_'.join(names)}": (
        lambda names=names: StackingClassifier(
            estimators=[(n, clf()) for n, clf in BASE.items() if n in names],
            final_estimator=LinearSVC(C=0.1, max_iter=2000, random_state=42),
            passthrough=False,
            n_jobs=-1,
        )
    )
    for r in range(2, len(BASE) + 1)
    for names in combinations(BASE.keys(), r)
}

registry = {**BASE, **_voting_entries, **_stacking_entries}

class FeatureExtractor():
    model_name:str 
    clf_name:str
    device:str 
    vision_model:object 
    train_loader:object
    val_loader:object
    test_loader:object
    transform:object

    def __init__(self, device:str, model_name:str, model_path:str, dropout=0.0):
        self.model_name = model_name
        self.transform = _get_transform(model_name)
        self.vision_model = build_vision_model(model_name=model_name, dropout=dropout, model_path=model_path)
        self.device = device

        if hasattr(self.vision_model, "fc"):                  
            self.vision_model.fc = torch.nn.Identity()
        elif hasattr(self.vision_model, "head"):              
            self.vision_model.head = torch.nn.Identity()
        elif hasattr(self.vision_model, "classifier"):        
            self.vision_model.classifier = torch.nn.Identity()

        self.vision_model.to(device)
        self.vision_model.eval()

    def set_loaders(self, dirs:dict,batch_size:int, num_workers:int):
        self.train_loader, self.val_loader, self.test_loader = _make_loaders(dirs, batch_size=batch_size, num_workers=num_workers, transform=self.transform)

    def _extract_features(self, loader:DataLoader) -> tuple:
        use_amp = _use_amp(self.model_name)
        feats, labels = [], []

        for images, batch_labels in tqdm(loader, desc="Extracting features"):
            images = images.to(self.device)
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.vision_model(images)
                else:
                    out = self.vision_model(images)

            if hasattr(out, "logits"):          
                out = out.logits
            
            feats.append(out.cpu())
            labels.append(batch_labels)

        return torch.cat(feats).numpy(), torch.cat(labels).numpy()
    
    def extract_and_save(self, save_dir: str, dataset_name: str, type_img: str):
        """Extrai features dos três splits e salva em disco."""
        os.makedirs(save_dir, exist_ok=True)
        for split, loader in [("train", self.train_loader), ("val", self.val_loader), ("test", self.test_loader)]:
            feat_path = os.path.join(save_dir, f"{split}-{self.model_name}-{dataset_name}-{type_img}.npz")
            if os.path.exists(feat_path):
                print(f"[SKIP] {feat_path} já existe.")
                continue
            X, y = self._extract_features(loader)
            np.savez(feat_path, X=X, y=y)
            print(f"[SAVED] {feat_path}")

    @staticmethod
    def load_features(save_dir: str, model_name: str, dataset_name: str, type_img: str):
        """Carrega features salvas para val e test."""
        def _load(split):
            path = os.path.join(save_dir, f"{split}-{model_name}-{dataset_name}-{type_img}.npz")
            data = np.load(path)
            return data["X"], data["y"]
        return _load("val"), _load("test")


            