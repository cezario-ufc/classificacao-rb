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
N_SPLITS = 10

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

IMG_SIZE = 512
USE_AUGMENT = True
BATCH_SIZE = 16
NUM_WORKERS = 8
EPOCHS_GRIDSEARCH = 3
EPOCHS_FINAL = 15
PARAM_GRID = {
    "lr": [1e-3, 1e-4],
    "weight_decay": [0.0, 1e-4],
}

USE_CHANNEL_DECOMP = False
CHANNEL_DECOMP_STATS = {
    "ddr": {
    "mean": [
      0.5316913323858822,
      0.22142141178307387,
      0.030600858964910185
    ],
    "std": [
      0.06116079606011062,
      0.21665438841917353,
      0.059284683854857426
    ],
    "n_images": 12522
  },
    "mesidor": {
        "mean": [0.5777163802016626, 0.1938902037452338, 0.03587220290509168],
        "std":  [0.104375319229332,  0.2020459362521818, 0.06907134685343622],
        "n_images": 1200,
    },
}
