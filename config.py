import torch

CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 100,
    "warmup_epochs": 5,
    "img_size": (256, 256),
    "num_classes": 21,
    "data_path": "./data",

    # EPIC
    "apply_epic": False,
    "purity_weight": 0.1,
    "wandb_project": "epic-voc-segmentation"
}
