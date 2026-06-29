#!/bin/bash
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
if [ "$VIRTUAL_ENV" != "$(pwd)/$VENV_DIR" ]; then
    echo "O ambiente '$VENV_DIR' não está ativado no momento."
    if [ -d "$VENV_DIR" ]; then
        echo "Diretório '$VENV_DIR' encontrado. Ativando o ambiente..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Ambiente '$VENV_DIR' não existe. Criando agora..."
        python3 -m venv "$VENV_DIR"
        echo "Ativando o ambiente recém-criado..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip > /dev/null 2>&1
        if [ -f "$REQUIREMENTS" ]; then
            echo "Arquivo $REQUIREMENTS encontrado. Instalando dependências..."
            pip install -r "$REQUIREMENTS"
        else
            echo "Aviso: Nenhum arquivo '$REQUIREMENTS' encontrado. Nenhuma dependência extra foi instalada."
        fi
    fi
    echo "Pronto! Ambiente ativado e configurado."
else
    echo "Tudo certo! O ambiente '$VENV_DIR' já está ativado e pronto para uso."
fi

MODELS=("ViTB16" "ConViT" "FastViT" "EfficientViT-B0" "DeiT-Tiny" "Swin-Tiny")
HTTP_DATASETS=("CSIC-2010" "FWAF" "HTTP-PARAMS")
TYPE_IMGS=("GASF" "GADF" "RPLOT" "SEQ")

IMAGES_ROOT="images"
MODEL_DIR="saved_models"
FEATURES_DIR="features"
RESULTS_CSV="results/ensemble_results.csv"

echo "========================================================="
echo " Iniciando bateria de experimentos - Ensemble Heads"
echo "========================================================="

for DATASET in "${HTTP_DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Processando Dataset: $DATASET"
    echo "---------------------------------------------------------"
    for MODEL in "${MODELS[@]}"; do
        for TYPE in "${TYPE_IMGS[@]}"; do
            echo " -> Modelo: $MODEL | Dataset: $DATASET | Tipo: $TYPE"
            python training/test_ensemble.py \
                --results_csv "$RESULTS_CSV" \
                --model_dir   "$MODEL_DIR" \
                --model_name  "$MODEL" \
                --type_img    "$TYPE" \
                --dataset     "$DATASET" \
                --image_dir   "$IMAGES_ROOT/$DATASET" \
                --features_dir "$FEATURES_DIR" \
                --batch_size  32 \
                --num_workers 8
            echo " -> [OK] $MODEL | $DATASET | $TYPE finalizado!"
            echo ""
        done
    done
done

echo "========================================================="
echo " Todos os experimentos ensemble foram concluídos!"
echo "========================================================="
