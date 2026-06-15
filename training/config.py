ROOT = f"../images/CSIC-2010"

DATA_DIR_BERT = {
    "train": f"{ROOT}/BERT-multilayer-3ch/train",
    "val":   f"{ROOT}/BERT-multilayer-3ch/val",
    "test":  f"{ROOT}/BERT-multilayer-3ch/test",
}

DATA_DIR_CWT = {
    "train": f"{ROOT}/CWT/train",
    "val":   f"{ROOT}/CWT/val",
    "test":  f"{ROOT}/CWT/test",
}
DATA_DIR_DI = {
    "train": f"{ROOT}/DeepInsight/train",
    "val":   f"{ROOT}/DeepInsight/val",
    "test":  f"{ROOT}/DeepInsight/test",
}
DATA_DIR_RGB = {
    "train": f"{ROOT}/RGB Combined/train",
    "val":   f"{ROOT}/RGB Combined/val",
    "test":  f"{ROOT}/RGB Combined/test",
}

DATA_DIR_GASF = {
    "train": f"{ROOT}/GASF_state0/train",
    "val":   f"{ROOT}/GASF_state0/val",
    "test":  f"{ROOT}/GASF_state0/test",
}

DATA_DIR_GADF = {
    "train": f"{ROOT}/GADF_state0/train",
    "val":   f"{ROOT}/GADF_state0/val",
    "test":  f"{ROOT}/GADF_state0/test",
}
DATA_DIR_RPLOT = {
    "train": f"{ROOT}/RPLOT_state0/train",
    "val":   f"{ROOT}/RPLOT_state0/val",
    "test":  f"{ROOT}/RPLOT_state0/test",
}

DATA_DIR_SEQ = {
    "train": f"{ROOT}/SEQ_state0/train",
    "val":   f"{ROOT}/SEQ_state0/val",
    "test":  f"{ROOT}/SEQ_state0/test",
}

DATA_DIR_MKF = {
    "train": f"{ROOT}/MKF/train",
    "val":   f"{ROOT}/MKF/val",
    "test":  f"{ROOT}/MKF/test",
}

DATA_DIR_MKF_C= {
    "train": f"{ROOT}/MKF-Combined/train",
    "val":   f"{ROOT}/MKF-Combined/val",
    "test":  f"{ROOT}/MKF-Combined/test",
}

#DATA_DIRS = [[DATA_DIR_RGB],[DATA_DIR_MKF],[DATA_DIR_MKF_C],[DATA_DIR_GASF],[DATA_DIR_GADF],[DATA_DIR_RPLOT]]
#DATA_DIRS = [[DATA_DIR_BERT]]
DATA_DIRS = [[DATA_DIR_GASF],[DATA_DIR_GADF],[DATA_DIR_RPLOT],[DATA_DIR_SEQ]]
BATCH_SIZE   = 16
P            = 0.35
WEIGHT_DECAY = 0.05
NUM_SAMPLES  = 70000
NUM_EPOCHS   = 100
NUM_DATASETS = 1
RESOLUTION   = 42
