# coding=utf-8
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader 
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
import timm
import time
import os
import torch.backends.cudnn as cudnn
import config 
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from myutils import train, test
from transformers import ViTForImageClassification, SwinForImageClassification,AutoImageProcessor
import argparse
from MiniCNN import MiniCNN
from CNN import ConvolutionalNeuralNetwork, build_model

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true", help="Congelar ou não camadas do modelo")
parser.add_argument("--epochs",type=int, default=50, help="Número de épocas")
parser.add_argument('--model', type=str, default='ResNet50', 
                    choices=['ResNet18', 'ResNet50', 'ConViT', 'ConvNext-Nano', 'ViTB16', 'Swin-Tiny', 'HybriDet', 'FastViT', 'MiniCNN', 'CustomCNN'], 
                    help='Escolha o modelo de visão ')

args = parser.parse_args()
#image_names = ["RGB", "MKF", "MKF-C", "GASF", "GADF", "RPLOT"]
image_names = ["GASF", "GADF", "RPLOT", "SEQ"]
#image_names = ["BERT"]
ig_count=0
model_name = args.model


save_dir = f"../saved_models/{model_name}/"
test_dataset_dir = f"../results/general_{model_name}.csv"

num_epochs = args.epochs
for data_dir in config.DATA_DIRS:
    for i in range(config.NUM_DATASETS):

        if model_name in ["MinICNN", "CustomCNN"]:
            if "BERT" in image_names[ig_count]:
                res = 32
            else:
                res = 42
            transform = transforms.Compose([
                transforms.Resize((res, res)), 
                transforms.ToTensor(),         
            ])
        else:
            res = 224
            transform = transforms.Compose([
                transforms.Resize((res, res)), 
                transforms.ToTensor(),         
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


        train_dataset = datasets.ImageFolder(root = data_dir[i]["train"], transform=transform)
        val_dataset = datasets.ImageFolder(root = data_dir[i]["val"], transform=transform)
        test_dataset = datasets.ImageFolder(root = data_dir[i]["test"], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size = config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)
        
        print("Dados carregados com sucesso...")

        save_model_name = f"{model_name}{i}-{image_names[ig_count]}-42-CSIC-2010"

        if model_name == "ResNet50":
            model = models.resnet50(weights='DEFAULT')
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
            nn.Dropout(config.P),
            nn.Linear(num_features, 2)
            )

        elif model_name == "CustomCNN":
            arch = {"name": "mid_3b_b",     "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, True)]}
            fc_dims = [512]
            model = build_model(arch=arch, fc_dims=fc_dims, dropout=0.5, use_bn=True)

        elif model_name == "ResNet18":
            model = models.resnet18(weights="DEFAULT")
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
            nn.Dropout(config.P),
            nn.Linear(num_features, 2)
            )

        elif model_name == "ConViT":
            model = timm.create_model('convit_tiny', pretrained=True, num_classes=2)

        elif model_name == "HybriDet":
            model = timm.create_model('vit_tiny_r_s16_p8_224', pretrained=True, num_classes=2)

        elif model_name == "FastViT":
            model = timm.create_model('fastvit_t8', pretrained=True, num_classes=2)
            
        elif model_name == "ConvNext-Nano":
            model = timm.create_model('convnext_nano',pretrained=True,num_classes=2,in_chans=3)

        elif model_name == "ViTB16":
            model = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224-in21k",  
                    num_labels=2,  
        )
        elif model_name == "Swin-Tiny":
            model = SwinForImageClassification.from_pretrained(
                    "microsoft/swin-tiny-patch4-window7-224",  
                    num_labels=2,
                    ignore_mismatched_sizes=True  
        )
        
        elif model_name == "MiniCNN":
            model = MiniCNN(num_classes=2)
        
        
        if model_name == "MiniCNN":
            requires = True
            print(train_dataset.classes)       # deve ser exatamente ['benign', 'malicious'] ou similar
            print(train_dataset.class_to_idx)  # deve ter só 2 entradas
            print(set(train_dataset.targets))
        
        if args.tf:
            requires = False
        else:
            requires = True

        for params in model.parameters():
            params.requires_grad = requires

        if model_name == "ResNet50" or model_name == "ResNet18":    
            for params in model.fc.parameters():
                params.requires_grad = True
        elif model_name in ["ConvNext-Nano", "ConViT", "HybriDet", "FastViT"]:
            for params in model.head.parameters():
                params.requires_grad = True
            
        elif model_name == "ViTB16" or model_name == "Swin-Tiny":
            for params in model.classifier.parameters():
                params.requires_grad = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
            
        model = model.to(device)
        
        best_train_acc, best_epoch= train(model, num_epochs, train_loader, val_loader, output_dir=save_dir, model_name=save_model_name, device=device)
        checkpoint_filename = f"{save_model_name}_{num_epochs}.pth"
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

        loss, acc, prec, rec, f1, auc = test(model, test_loader,model_name, device=device)

        print(f'Melhor acurácia de treinamento: {best_train_acc} atingida com {best_epoch} épocas')

        data = {"Image_Dataset": f"144-TEST-{image_names[ig_count]}{i}-notf-CSIC-2010",
            "Model": model_name, 
            "Epochs": num_epochs,
            "Test_Loss":loss,
            "Test_Acuracia": acc, 
            "Test_Precisao": prec, 
            "Test_Recall": rec, 
            "Test_F1-Score": f1,
            "Test_ROC-AUC": auc,
            "Best_Acc_Train":best_train_acc,
            "Best_Epoch_Acc":best_epoch,
            "Num_Samples":config.NUM_SAMPLES,
            "Dropout":config.P,
           "Resolution":config.RESOLUTION,
            "Data Normalization": 'Yes', 
            }

        test_df = pd.DataFrame([data])

        if os.path.exists(test_dataset_dir):
            test_df.to_csv(test_dataset_dir, mode='a', header=False, index=False)
        else:
            test_df.to_csv(test_dataset_dir, mode='w', header=True, index=False)
    ig_count+=1
