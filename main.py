import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from config import CONFIG
from dataset import VOCDatasetManager
from model import get_deeplab_model
from trainer import train_one_epoch
from utils import visualize_prediction, save_checkpoint

def main():
    epic_path = "epic_checkpoint.pth"
    base_path = "deeplab_checkpoint.pth"
    apply_epic = CONFIG.get("apply_epic", False)

    save_path = epic_path if apply_epic else base_path

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    dm = VOCDatasetManager(CONFIG)
    train_loader, test_loader = dm.get_dataloaders()

    model = get_deeplab_model(CONFIG["num_classes"], apply_epic=apply_epic).to(CONFIG["device"])


    if apply_epic:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[4].M.requires_grad = True

    loaded = False

    if apply_epic:
        if os.path.exists(epic_path):
            print(f"Loading EPIC: {epic_path}")
            state_dict = torch.load(epic_path, map_location=CONFIG["device"])
            model.load_state_dict(state_dict)
            loaded = True
        elif os.path.exists(base_path):
            print(f"No EPIC checkpoint found. Trying to load DeepLab checkpoint: {base_path}...")
            state_dict = torch.load(base_path, map_location=CONFIG["device"])

            epic_state_dict = {}
            for k, v in state_dict.items():
                if k == "classifier.4.weight":
                    epic_state_dict["classifier.4.original_weight"] = v
                elif k == "classifier.4.bias":
                    epic_state_dict["classifier.4.bias"] = v
                else:
                    epic_state_dict[k] = v

            model.load_state_dict(epic_state_dict, strict=False)
            loaded = True
    else:
        if os.path.exists(base_path):
            print(f"Loading base model: {base_path}")
            state_dict = torch.load(base_path, map_location=CONFIG["device"])
            model.load_state_dict(state_dict)
            loaded = True

    if not loaded:
        print("No checkpoint found.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])

    warmup_epochs = CONFIG["warmup_epochs"]
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / (warmup_epochs + 1e-8))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    print(f"Start: {CONFIG['device']} | EPIC: {apply_epic} | Saving to: {save_path}")

    try:
        for epoch in range(CONFIG["epochs"]):
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                CONFIG["device"], epoch + 1, CONFIG
            )

            current_lr = optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch + 1,
                "train/avg_epoch_loss": avg_loss,
                "train/learning_rate": current_lr
            })

            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

            scheduler.step()

    except KeyboardInterrupt:
        print("Training stopped. Saving")

    visualize_prediction(model, test_loader, CONFIG["device"])

    save_checkpoint(model, save_path)

    print(f"Model saved as {save_path}.")
    wandb.finish()

if __name__ == "__main__":
    main()