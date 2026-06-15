# config.py


def _dirs(name: str, root: str) -> dict:
    return {split: f"{root}/{name}/{split}" for split in ("train", "val", "test")}


def get_datasets(root: str) -> list:
    return [
        {"name": "GASF", "dirs": _dirs("GASF_state0", root)},
        {"name": "GADF", "dirs": _dirs("GADF_state0", root)},
        {"name": "RPLOT", "dirs": _dirs("RPLOT_state0", root)},
        {"name": "SEQ", "dirs": _dirs("SEQ_state0", root)},
    ]


P = 0.35
WEIGHT_DECAY = 0.05
NUM_SAMPLES = 70000
NUM_EPOCHS = 100
RESOLUTION = 42
