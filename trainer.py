import torch
import torch.nn.functional as F


def compute_purity_loss(features):
    B, C, H, W = features.shape
    acts = F.relu(features).view(B, C, -1)

    epsilon = 1e-8
    acts_sum = acts.sum(dim=-1, keepdim=True) + epsilon
    probs = acts / acts_sum

    entropy = - (probs * torch.log(probs + epsilon)).sum(dim=-1)

    active_mask = (acts.sum(dim=-1) > 0).float()
    valid_channels = active_mask.sum(dim=-1).clamp(min=1.0)

    mean_entropy = (entropy * active_mask).sum(dim=-1) / valid_channels
    return mean_entropy.mean()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, config):
    if config.get("apply_epic", False):
        model.eval()
        if hasattr(model.classifier[4], "train"):
            model.classifier[4].train()
    else:
        model.train()
    total_loss = 0

    from tqdm import tqdm
    import wandb
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")

    for batch_idx, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)['out']

        if config.get("apply_epic", False):
            features = model.classifier[4].last_disentangled_features
            M = model.classifier[4].M
            purity_loss = compute_purity_loss(features)

            identity = torch.eye(M.size(0), device=M.device)
            ortho_loss = torch.norm(torch.matmul(M.t(), M) - identity)

            loss = (config["purity_weight"] * purity_loss) + (0.1 * ortho_loss)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)