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
            echo "Aviso: Nenhum arquivo '$REQUIREMENTS' encontrado na pasta. Nenhuma dependência extra foi instalada."
        fi
    fi
    echo "Pronto! Ambiente ativado e configurado."
else
    echo "Tudo certo! O ambiente '$VENV_DIR' já está ativado e pronto para uso."
fi

MODELS=("ViTB16" "ConViT" "FastViT" "EfficientViT-B0" "DeiT-Tiny" "Swin-Tiny")
HTTP_DATASETS=("CSIC-2010" "FWAF" "HTTP-PARAMS")
IMAGES_ROOT=""

echo "========================================================="
echo " Iniciando bateria de experimentos - Modelos ViT"
echo "========================================================="

for DATASET in "${HTTP_DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Processando Dataset: $DATASET"
    echo "---------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
        echo " -> Treinando Modelo: $MODEL no Dataset: $DATASET"

        python training/general_test.py \
            --model "$MODEL" \
            --epochs 30 \
            --root "$IMAGES_ROOT/$DATASET" \
            --num_workers 8 \
            --batch_size 16

        echo " -> [OK] $MODEL no $DATASET finalizado!"
        echo ""
    done
done

echo "========================================================="
echo " Todos os experimentos foram concluídos com sucesso!"
echo "========================================================="
