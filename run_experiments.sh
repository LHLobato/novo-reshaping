#!/bin/bash

set -e

MODELS=("ViTB16" "ConViT" "FastViT" "MaxViT-Tiny" "EfficientViT-B0" "DeiT-Tiny" "DeiT-Small" "Swin-Tiny")
DATASETS=("CSIC-2010" "FWAF" "HTTP-PARAMS")

DATA_ROOT="../images"

echo "========================================================="
echo " Iniciando bateria de experimentos - Modelos ViT"
echo "========================================================="

for DATASET in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Processando Dataset: $DATASET"
    echo "---------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
        echo " -> Treinando Modelo: $MODEL no Dataset: $DATASET"

        python training/general_test.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --epochs 30 \
            --root "$DATA_ROOT" \
            --num_workers 8 \
            --batch_size 16

        echo " -> [OK] $MODEL no $DATASET finalizado!"
        echo ""
    done
done

echo "========================================================="
echo " Todos os experimentos foram concluídos com sucesso!"
echo "========================================================="
