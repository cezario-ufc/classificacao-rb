from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DDR_DIR = DATA_RAW / "oia-ddr" / "DDR-dataset" / "DR_grading"
MESSIDOR_DIR = DATA_RAW / "mesidor"
IDRID_DIR = DATA_RAW / "idrid" / "B. Disease Grading" / "B. Disease Grading"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLASSIFICATION_INDEX_CSV = OUTPUTS_DIR / "classification_index.csv"
SPLITS_CSV = OUTPUTS_DIR / "splits.csv"

VAL_RATIO = 0.15
N_SPLITS = 5

CLASSES_TO_DROP = [5]
DDR_LABEL_REMAP = {4: 3}
IDRID_LABEL_REMAP = {4: 3}
NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS_GRIDSEARCH = 3
EPOCHS_FINAL = 5
PARAM_GRID = {
    "lr": [1e-3, 1e-4],
    "weight_decay": [0.0, 1e-4],
}
