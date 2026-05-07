import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from config import CONFIG
from dataset import VOCDatasetManager
from model import get_deeplab_model


def load_trained_model(checkpoint_path):
    model = get_deeplab_model(CONFIG["num_classes"], apply_epic=True)
    state_dict = torch.load(checkpoint_path, map_location=CONFIG["device"])
    model.load_state_dict(state_dict)
    model.to(CONFIG["device"])
    model.eval()
    return model


def get_img_for_vis(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img = inv_normalize(tensor).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


def get_max_activation_patch(feature_map, img_size=(256, 256), patch_size=40):
    h, w = feature_map.shape
    idx = np.argmax(feature_map)
    y, x = np.unravel_index(idx, (h, w))

    scale_y, scale_x = img_size[0] / h, img_size[1] / w
    center_y, center_x = int((y + 0.5) * scale_y), int((x + 0.5) * scale_x)

    top_left_x = np.clip(center_x - patch_size // 2, 0, img_size[1] - patch_size)
    top_left_y = np.clip(center_y - patch_size // 2, 0, img_size[0] - patch_size)
    return top_left_x, top_left_y


def find_train_exemplars(model, train_loader, channel_indices, num_samples=500, top_k=4):
    model.eval()
    device = CONFIG["device"]
    exemplars = {idx: [] for idx in channel_indices}

    processed_samples = 0
    with torch.no_grad():
        for images, _ in train_loader:
            if processed_samples >= num_samples:
                break

            images = images.to(device)
            _ = model(images)
            features = model.classifier[4].last_disentangled_features

            batch_scores = torch.amax(features[:, channel_indices, :, :], dim=(2, 3))
            scores_cpu = batch_scores.cpu().numpy()

            for local_idx, c_idx in enumerate(channel_indices):
                for b_idx in range(images.size(0)):
                    score = scores_cpu[b_idx, local_idx]

                    current_list = exemplars[c_idx]
                    if len(current_list) < top_k or score > current_list[-1][0]:
                        img_cpu = images[b_idx].cpu().clone()
                        feat_map_cpu = features[b_idx, c_idx].cpu().numpy()

                        current_list.append((score, img_cpu, feat_map_cpu))
                        current_list.sort(key=lambda x: x[0], reverse=True)
                        exemplars[c_idx] = current_list[:top_k]

            processed_samples += images.size(0)

    return exemplars


def visualize_full_epic_analysis(test_img, gt_mask, pred_mask, features, exemplars_data, top4_channels, patch_size=40,
                                 save_path="output.png"):
    img_vis = get_img_for_vis(test_img)
    img_size = img_vis.shape[:2]

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(5, 5, hspace=0.4, wspace=0.1)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_title("Input Image", fontweight='bold')
    ax_input.imshow(img_vis)
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.set_title("Ground Truth", fontweight='bold')
    ax_gt.imshow(gt_mask.cpu().numpy(), cmap='viridis')
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_pred.set_title("Prediction (Mask)", fontweight='bold')
    ax_pred.imshow(pred_mask, cmap='viridis')

    for row_idx, c_idx in enumerate(top4_channels):
        grid_row = row_idx + 1

        ax_test = fig.add_subplot(gs[grid_row, 0])
        ax_test.imshow(img_vis)
        tx, ty = get_max_activation_patch(features[0, c_idx].cpu().numpy(), img_size, patch_size)
        ax_test.add_patch(
            patches.Rectangle((tx, ty), patch_size, patch_size, linewidth=3, edgecolor='red', facecolor='none'))
        ax_test.set_ylabel(f"Channel {c_idx}", fontsize=14, fontweight='bold')

        train_list = exemplars_data.get(c_idx, [])
        for col_idx in range(1, 5):
            ax_ex = fig.add_subplot(gs[grid_row, col_idx])
            if col_idx - 1 < len(train_list):
                _, ex_tensor, ex_f_map = train_list[col_idx - 1]
                ex_img = get_img_for_vis(ex_tensor)
                ax_ex.imshow(ex_img)
                ex_x, ex_y = get_max_activation_patch(ex_f_map, ex_img.shape[:2], patch_size)
                ax_ex.add_patch(
                    patches.Rectangle((ex_x, ex_y), patch_size, patch_size, linewidth=3, edgecolor='#00FF00',
                                      facecolor='none'))
            else:
                ax_ex.text(0.5, 0.5, "N/A", ha='center')

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("EPIC: Mask-Targeted Purity Analysis", fontsize=22, y=0.95)

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Visualizastions saved as: {save_path}")


def run_prediction_flow(checkpoint_path=None, num_images_to_process=5):
    if checkpoint_path is None:
        checkpoint_path = "epic_checkpoint.pth" if CONFIG.get("apply_epic", False) else "deeplab_checkpoint.pth"

    print(f"Using model: {checkpoint_path}")
    model = load_trained_model(checkpoint_path)
    dm = VOCDatasetManager(CONFIG)
    train_loader, test_loader = dm.get_dataloaders()

    output_dir = "epic_results"
    os.makedirs(output_dir, exist_ok=True)

    dataset_size = len(test_loader.dataset)
    random_indices = random.sample(range(dataset_size), min(num_images_to_process, dataset_size))

    for iteration, idx in enumerate(tqdm(random_indices, desc="Images processing")):
        img_tensor, gt_mask = test_loader.dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(CONFIG["device"])

        with torch.no_grad():
            output = model(input_batch)['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            epic_features = model.classifier[4].last_disentangled_features

        _, _, h_f, w_f = epic_features.shape
        pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float().to(CONFIG["device"])
        mask_rescaled = F.interpolate(pred_mask_tensor, size=(h_f, w_f), mode='nearest').squeeze()
        object_mask = ((mask_rescaled > 0) & (mask_rescaled < 255)).float()

        if object_mask.sum() == 0:
            channel_relevance = epic_features.squeeze(0).mean(dim=(1, 2))
        else:
            masked_features = epic_features.squeeze(0) * object_mask
            channel_relevance = masked_features.sum(dim=(1, 2)) / (object_mask.sum() + 1e-8)

        features_2d = F.relu(epic_features.squeeze(0)).view(epic_features.shape[1], -1)
        acts_sum = features_2d.sum(dim=-1, keepdim=True) + 1e-8
        probs = features_2d / acts_sum
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)

        combined_score = channel_relevance / (entropy + 1e-4)
        top4_channels = torch.topk(combined_score, 4).indices.cpu().numpy()

        exemplars_data = find_train_exemplars(model, train_loader, top4_channels, num_samples=100)

        save_filename = os.path.join(output_dir, f"epic_analysis_{iteration + 1}_img_{idx}.png")

        visualize_full_epic_analysis(img_tensor, gt_mask, pred_mask, epic_features, exemplars_data, top4_channels,
                                     save_path=save_filename)

    print(f"Done. Saved as: '{output_dir}'.")


if __name__ == "__main__":
    run_prediction_flow()