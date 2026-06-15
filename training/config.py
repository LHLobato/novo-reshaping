# config.py
ROOT = "../images/CSIC-2010"


def _dirs(name: str) -> dict:
    return {split: f"{ROOT}/{name}/{split}" for split in ("train", "val", "test")}


def get_datasets(root: str) -> list:
    return [
        {"name": "GASF", "dirs": _dirs("GASF_state0")},
        {"name": "GADF", "dirs": _dirs("GADF_state0")},
        {"name": "RPLOT", "dirs": _dirs("RPLOT_state0")},
        {"name": "SEQ", "dirs": _dirs("SEQ_state0")},
    ]


P = 0.35
WEIGHT_DECAY = 0.05
NUM_SAMPLES = 70000
NUM_EPOCHS = 100
RESOLUTION = 42
