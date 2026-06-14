import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from config import CONFIG
from dataset import VOCDatasetManager
from model import get_deeplab_model
from trainer import train_one_epoch, build_prototype_loader
from utils import save_checkpoint


def main():
    results_dir = CONFIG.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    epic_path = os.path.join(results_dir, "epic_checkpoint.pth")
    base_path = os.path.join(results_dir, "deeplab_checkpoint.pth")

    apply_epic = CONFIG.get("apply_epic", False)
    train_mode = CONFIG.get("train_mode", "classifier")

    save_path = epic_path if apply_epic else base_path

    wandb.init(project=CONFIG.get("wandb_project", "epic"), config=CONFIG)

    dm = VOCDatasetManager(CONFIG)
    train_loader, test_loader = dm.get_dataloaders()

    model = get_deeplab_model(CONFIG["num_classes"], train_mode=train_mode, apply_epic=apply_epic).to(CONFIG["device"])

    if apply_epic:
        if os.path.exists(epic_path):
            state_dict = torch.load(epic_path, map_location=CONFIG["device"])
            model.load_state_dict(state_dict)
        elif os.path.exists(base_path):
            state_dict = torch.load(base_path, map_location=CONFIG["device"])
            if "classifier.4.weight" in state_dict:
                state_dict["classifier.4.original_weight"] = state_dict.pop("classifier.4.weight")
            model.load_state_dict(state_dict, strict=False)
    else:
        if os.path.exists(base_path):
            state_dict = torch.load(base_path, map_location=CONFIG["device"])
            model.load_state_dict(state_dict)

    if apply_epic:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"],
                               weight_decay=1e-4)

    total_epochs = CONFIG["epochs"]
    warmup_epochs = CONFIG.get("warmup_epochs", 0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1e-8)
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    try:
        current_loader = train_loader
        num_prototypes = 100

        for epoch in range(total_epochs):

            if apply_epic and epoch % 2 == 0:
                if total_epochs > 1:
                    num_prototypes = int(100 - (100 - 5) * (epoch / (total_epochs - 1)))
                else:
                    num_prototypes = 5

                num_prototypes = max(5, num_prototypes)

                current_loader = build_prototype_loader(
                    model=model,
                    base_loader=train_loader,
                    num_prototypes=num_prototypes,
                    device=CONFIG["device"]
                )

            avg_loss = train_one_epoch(
                model, current_loader, optimizer, criterion,
                CONFIG["device"], epoch + 1, CONFIG
            )

            current_lr = optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch + 1,
                "train/avg_epoch_loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/prototypes_per_channel": num_prototypes if apply_epic else -1
            })

            scheduler.step()
            save_checkpoint(model, filename=save_path)

    except KeyboardInterrupt:
        save_checkpoint(model, filename=save_path)


if __name__ == "__main__":
    main()