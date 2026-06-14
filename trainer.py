import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def compute_purity_loss(features):
    B, C, H, W = features.shape
    features_flat = features.view(B, C, -1)
    epsilon = 1e-8

    probs = F.softmax(features_flat, dim=-1)
    entropy = - (probs * torch.log(probs + epsilon)).sum(dim=-1)

    channel_max = torch.abs(features_flat).max(dim=-1)[0]
    channel_weights = channel_max / (channel_max.sum(dim=-1, keepdim=True) + epsilon)

    weighted_entropy = (entropy * channel_weights).sum(dim=-1)

    return weighted_entropy.mean()


def build_prototype_loader(model, base_loader, num_prototypes, device):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for images, _ in base_loader:
            images = images.to(device)
            _ = model(images)
            features = model.classifier[4].last_disentangled_features
            B, C, H, W = features.shape
            f_flat = features.view(B, C, -1)

            probs = F.softmax(f_flat, dim=-1)
            entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_acts = torch.abs(f_flat).max(dim=-1)[0]

            purity_score = max_acts / (entropy + 1e-4)
            all_scores.append(purity_score.cpu())

    all_scores = torch.cat(all_scores, dim=0)
    num_images = all_scores.shape[0]
    num_channels = all_scores.shape[1]

    selected_indices = set()
    for c in range(num_channels):
        channel_scores = all_scores[:, c]
        top_k_indices = torch.topk(channel_scores, min(num_prototypes, num_images)).indices.numpy()
        selected_indices.update(top_k_indices)

    selected_indices = list(selected_indices)
    prototype_subset = Subset(base_loader.dataset, selected_indices)

    return DataLoader(
        prototype_subset,
        batch_size=base_loader.batch_size,
        shuffle=True,
        drop_last=True
    )


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, config):
    is_epic = config.get("apply_epic", False)

    if is_epic:
        model.eval()
        if hasattr(model.classifier[4], "train"):
            model.classifier[4].train()
    else:
        model.train()

    total_loss = 0
    from tqdm import tqdm
    pbar = tqdm(enumerate(loader), total=len(loader))

    for batch_idx, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if is_epic:
            _ = model(images)
            features = model.classifier[4].last_disentangled_features
            loss = compute_purity_loss(features)
        else:
            outputs = model(images)['out']
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)