import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from transformers import get_cosine_schedule_with_warmup
import os 
import time
import numpy as np
from tqdm import tqdm 


class ConvolutionalNeuralNetwork(nn.Module):
  def __init__(self, conv_layers: list, fc_layers: list):
      super().__init__()
      self.conv_block = nn.Sequential(*conv_layers)
      self.fc_block   = nn.Sequential(*fc_layers)

  def forward(self, x):
      x = self.conv_block(x)
      x = x.flatten(start_dim=1)
      x = self.fc_block(x)
      return x

def _infer_flat_size(conv_block: nn.Sequential, img_size: int, in_channels: int) -> int:
    """Passa um tensor dummy pela parte convolucional para descobrir o tamanho achatado."""
    dummy = torch.zeros(1, in_channels, img_size, img_size)
    with torch.no_grad():
        out = conv_block(dummy)
    return int(out.flatten(start_dim=1).shape[1])


def build_conv_layers(blocks: list, in_channels: int = 3, use_bn: bool = True) -> nn.Sequential:
    """
    Constrói a parte convolucional.

    blocks: lista de tuplas (out_channels, kernel_size, use_pool)
    """
    layers = []
    ch_in = in_channels
    for (ch_out, k, pool) in blocks:
        layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=k, padding=k // 2))
        if use_bn:
            layers.append(nn.BatchNorm2d(ch_out))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        ch_in = ch_out
    return nn.Sequential(*layers)


def build_fc_layers(flat_size: int, hidden_dims: list, dropout: float,
                    num_classes: int = 2) -> nn.Sequential:
    """
    Constrói a cabeça totalmente conectada.

    hidden_dims: lista de inteiros com os tamanhos das camadas ocultas.
    """
    layers = []
    in_dim = flat_size
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)

SEED          = 42
IMG_SIZE      = 42         
IN_CHANNELS   = 3
NUM_CLASSES = 2

def build_model(arch: dict, fc_dims: list, dropout: float,
                use_bn: bool = True) -> ConvolutionalNeuralNetwork:
    conv_block = build_conv_layers(arch["blocks"], IN_CHANNELS, use_bn)
    flat_size  = _infer_flat_size(conv_block, IMG_SIZE, IN_CHANNELS)
    fc_block   = build_fc_layers(flat_size, fc_dims, dropout, NUM_CLASSES)
    return ConvolutionalNeuralNetwork(list(conv_block), list(fc_block))




def train(model, num_epochs, train_loader, val_loader, output_dir, model_name, lr, weight_decay, device='cuda',accumulation_steps=1): 
    start_time = time.time()
    patience = 5
    print(f'Iniciando treinamento...')
    
    curr_epoch = 0
    criterion = torch.nn.CrossEntropyLoss()
    

    cond1 = ('vit' in model_name.lower() or 'swin' in model_name.lower())
    cond2 = ('convnext' in model_name.lower())

    cond3 = ('coatnet' in model_name.lower() or 'maxvit' in model_name.lower())
    

    use_amp = True
    if cond3:
        print("!!! CoAtNet/MaxViT detectado: DESLIGANDO Mixed Precision (AMP) para estabilidade !!!")
        print("!!! Rodando em FP32 (Full Precision).")
        use_amp = False
    
    # Scaler só é instanciado se formos usar AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    condition = cond1 or cond2 or cond3
    
    if condition:
        if cond3:
            # Mantemos o LR baixo mesmo em FP32 por segurança
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.05)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(0.1 * num_epochs * len(train_loader)) // accumulation_steps,
            num_training_steps=(num_epochs * len(train_loader)) // accumulation_steps,
        )
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)

    best_epoch = 0
    best_accuracy = float('-inf')
    current_lr = 0
    patience_limit = 0
    
    optimizer.zero_grad()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        
        all_preds, all_labels, all_proba = [], [], []
        print(f'Epoch: {epoch + 1}....')

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if hasattr(outputs, 'logits'): logits = outputs.logits
                    else: logits = outputs
                    loss = criterion(logits, labels)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if condition: scheduler.step()

            else:
                # Caminho Seguro (FP32) - CoAtNet usa este
                outputs = model(images)
                if hasattr(outputs, 'logits'): logits = outputs.logits
                else: logits = outputs
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
                
                loss.backward() # Backward normal sem scaler
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if condition: scheduler.step()

            train_pred_class = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # --- CHECAGEM DE SEGURANÇA IMEDIATA ---
            prob_cpu = probabilities[:, 1].detach().cpu().numpy()
            

            if np.isnan(prob_cpu).any():
                print(f" [Aviso] Batch {i} gerou NaNs! Ignorando métricas deste batch.")
                continue 
            running_loss += loss.item() * accumulation_steps
            all_proba.extend(prob_cpu)
            all_preds.extend(train_pred_class.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_proba = np.nan_to_num(all_proba, nan=0.0)
        
        if len(all_labels) == 0:
            print("CRÍTICO: Todos os batches falharam. Abortando época.")
            break

        train_report_dict = classification_report(all_labels, all_preds, output_dict=True)
        train_accuracy = train_report_dict['accuracy']
        precision = train_report_dict['macro avg']['precision']
        recall    = train_report_dict['macro avg']['recall']
        f1        = train_report_dict['weighted avg']['f1-score']
        
        try:
            train_auc = roc_auc_score(all_labels, all_proba)
        except ValueError as e:
            print(f"Erro ao calcular AUC (provavelmente apenas 1 classe detectada): {e}")
            train_auc = 0.5

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        va_preds, va_labels, va_proba = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Validação também respeita a flag use_amp
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        if hasattr(outputs, 'logits'): logits = outputs.logits
                        else: logits = outputs
                        loss = criterion(logits, labels)
                else:
                    outputs = model(images)
                    if hasattr(outputs, 'logits'): logits = outputs.logits
                    else: logits = outputs
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                pred_class = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                
                prob_vals = probabilities[:, 1].detach().cpu().numpy()
                va_proba.extend(prob_vals)
                va_preds.extend(pred_class.detach().cpu().numpy())
                va_labels.extend(labels.cpu().numpy())
        
        # Blindagem da validação também
        va_proba = np.nan_to_num(va_proba, nan=0.0)

        val_report_dict = classification_report(va_labels, va_preds, output_dict=True)
        val_accuracy = val_report_dict['accuracy']
        val_precision_class_0 = val_report_dict['macro avg']['precision']
        val_recall_macro_avg = val_report_dict['macro avg']['recall']
        val_f1_weighted_avg = val_report_dict['weighted avg']['f1-score']
        try:
            val_auc = roc_auc_score(va_labels, va_proba)
        except:
            val_auc = 0.5

        if not condition:
            scheduler.step(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_limit = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_{num_epochs}.pth"))
        else:
            patience_limit+=1
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"| Train Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_accuracy:.4f} | ROC-AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_accuracy:.4f} | ROC-AUC: {val_auc:.4f}")
        
        if patience_limit >= patience:
            print("Early Stopping!")
            break
        curr_epoch+=1

    total_time = (time.time() - start_time) / 60
    print(f'Training Took: {total_time:.2f} minutes!')
    return best_accuracy, best_epoch

    #----Função de Teste do modelo.
def test(model, test_loader, model_name, device='cuda'):
    test_preds = []
    test_labels = []
    test_proba = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    # --- Lógica de Detecção (Igual ao Treino) ---
    cond1 = ('vit' in model_name.lower() or 'swin' in model_name.lower())

    cond3 = ('coatnet' in model_name.lower())
    

    use_amp = True
    if cond3:
        print("!!! Teste: CoAtNet/MaxViT detectado. Desligando AMP (FP32) para evitar NaN. !!!")
        use_amp = False

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testando"): 
            images, labels = images.to(device), labels.to(device)


            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    if cond1 or hasattr(outputs, 'logits'):
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    else:
                        logits = outputs
                    loss = criterion(logits, labels)
            else:

                outputs = model(images)
                if cond1 or hasattr(outputs, 'logits'):
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                else:
                    logits = outputs
                loss = criterion(logits, labels)

            test_loss += loss.item()
            test_pred_class = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            
            prob_vals = probabilities[:, 1].detach().cpu().numpy()
            test_proba.extend(prob_vals)
            test_preds.extend(test_pred_class.detach().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            

    test_proba = np.nan_to_num(test_proba, nan=0.0)


    if len(test_labels) > 0:
        test_report_dict = classification_report(test_labels, test_preds, output_dict=True)
        test_accuracy = test_report_dict['accuracy']
        test_precision_class_0 = test_report_dict['macro avg']['precision']
        test_recall_macro_avg = test_report_dict['macro avg']['recall']
        test_f1_weighted_avg = test_report_dict['weighted avg']['f1-score']
        
        try:
            test_auc = roc_auc_score(test_labels, test_proba)
        except ValueError:
            print("Erro ao calcular ROC-AUC no teste (provável NaN residual ou classe única). Definindo como 0.5")
            test_auc = 0.5
    else:
        print("Erro: DataLoader de teste vazio.")
        return 0, 0, 0, 0, 0, 0
        
    print(f"Test Loss: {test_loss / len(test_loader):.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
        f"Test Prec: {test_precision_class_0:.4f} | "
        f"Test Rec: {test_recall_macro_avg:.4f} | "
        f"Test F1: {test_f1_weighted_avg:.4f}|"
        f"Test ROC-AUC: {test_auc:.4f}")

    return test_loss / len(test_loader), test_accuracy, test_precision_class_0, test_recall_macro_avg, test_f1_weighted_avg, test_auc
