#!/bin/bash
set -e

MODELS=("ViTB16" "ConViT" "FastViT" "EfficientViT-B0" "DeiT-Tiny" "Swin-Tiny")
DATASETS=("CSIC-2010" "FWAF" "HTTP-PARAMS")
IMAGES_ROOT="../images"

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
