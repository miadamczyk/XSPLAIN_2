import torch

CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "lr": 1e-5,
    "epochs": 100,
    "warmup_epochs": 3,
    "img_size": (512, 512),
    "num_classes": 21,
    "data_path": "./data",
    "results_dir": "./results",
    "train_mode": "full",
    "apply_epic": False,
    "wandb_project": "epic-voc-segmentation"
}