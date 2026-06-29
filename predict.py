import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CONFIG
from dataset import VOCDatasetManager
from model import get_deeplab_model

VOC_CLASSES = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
    10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
    15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train",
    20: "tvmonitor"
}


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


def build_exemplar_database(model, dataset, save_path):
    model.eval()
    device = CONFIG["device"]
    all_scores = []

    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Building exemplar database"):
            images = images.to(device)
            _ = model(images)
            features = model.classifier[4].last_disentangled_features
            B, C, H, W = features.shape
            features_2d = features.view(B, C, -1)

            probs = F.softmax(features_2d, dim=-1)
            entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_acts = torch.abs(features_2d).max(dim=-1)[0]
            batch_scores = max_acts / (entropy + 1e-4)

            all_scores.append(batch_scores.cpu())

    scores_matrix = torch.cat(all_scores, dim=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(scores_matrix, save_path)
    return scores_matrix


def find_train_exemplars(model, dataset, scores_matrix, channel_indices, top_k=4):
    model.eval()
    device = CONFIG["device"]
    exemplars = {idx: [] for idx in channel_indices}

    for c_idx in channel_indices:
        channel_scores = scores_matrix[:, c_idx]
        topk_vals, topk_indices = torch.topk(channel_scores, top_k)

        for score, img_idx in zip(topk_vals, topk_indices):
            img_tensor, _ = dataset[img_idx.item()]
            img_batch = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                _ = model(img_batch)
                features = model.classifier[4].last_disentangled_features

            img_cpu = img_tensor.clone()
            feat_map_cpu = torch.abs(features[0, c_idx]).cpu().numpy()

            exemplars[c_idx].append((score.item(), img_cpu, feat_map_cpu))

        exemplars[c_idx].sort(key=lambda x: x[0], reverse=True)

    return exemplars


def visualize_full_epic_analysis(test_img, gt_mask, pred_mask, features, exemplars_data, top_channels, top_scores,
                                 chosen_classes, patch_size=40, save_path="output.png"):
    img_vis = get_img_for_vis(test_img)
    img_size = img_vis.shape[:2]
    num_rows = len(top_channels)

    H_feat, W_feat = features.shape[2], features.shape[3]
    pred_mask_tensor = torch.tensor(pred_mask, dtype=torch.float32)
    pred_mask_resized = F.interpolate(
        pred_mask_tensor.unsqueeze(0).unsqueeze(0),
        size=(H_feat, W_feat),
        mode='nearest'
    ).squeeze().numpy()

    fig = plt.figure(figsize=(22, 4 * (num_rows + 1)))
    gs = fig.add_gridspec(num_rows + 1, 6, hspace=0.4, wspace=0.1)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(img_vis)
    ax_input.set_title("Original Image")

    ax_mask_title = fig.add_subplot(gs[0, 1])
    ax_mask_title.axis('off')
    ax_mask_title.set_title("Isolated Class")

    ax_gt = fig.add_subplot(gs[0, 2])
    gt_vis = gt_mask.cpu().numpy().copy()

    gt_vis[gt_vis == 255] = 0

    ax_gt.imshow(gt_vis, cmap='viridis', vmin=0, vmax=20)
    ax_gt.set_title("Ground Truth")

    ax_pred = fig.add_subplot(gs[0, 3])

    ax_pred.imshow(pred_mask, cmap='viridis', vmin=0, vmax=20)
    ax_pred.set_title("Prediction")

    for row_idx, c_idx in enumerate(top_channels):
        grid_row = row_idx + 1
        cls_id = chosen_classes[row_idx]
        cls_name = VOC_CLASSES.get(cls_id, str(cls_id))

        ax_test = fig.add_subplot(gs[grid_row, 0])
        ax_test.imshow(img_vis)
        ax_test.set_title(f"{cls_name} (Ch: {c_idx})")

        abs_feature = torch.abs(features[0, c_idx]).cpu().numpy()

        masked_feature = np.copy(abs_feature)
        masked_feature[pred_mask_resized != cls_id] = -1.0

        tx, ty = get_max_activation_patch(masked_feature, img_size, patch_size)

        ax_test.add_patch(
            patches.Rectangle((tx, ty), patch_size, patch_size, linewidth=3, edgecolor='red', facecolor='none'))

        ax_masked = fig.add_subplot(gs[grid_row, 1])

        masked_img = np.zeros_like(img_vis)
        masked_img[:] = [68 / 255.0, 1 / 255.0, 84 / 255.0]

        mask_bool = (pred_mask == cls_id)
        masked_img[mask_bool] = img_vis[mask_bool]

        ax_masked.imshow(masked_img)

        train_list = exemplars_data.get(c_idx, [])
        for col_idx in range(1, 5):
            ax_ex = fig.add_subplot(gs[grid_row, col_idx + 1])
            if col_idx - 1 < len(train_list):
                _, ex_tensor, ex_f_map = train_list[col_idx - 1]
                ex_img = get_img_for_vis(ex_tensor)
                ax_ex.imshow(ex_img)

                ex_x, ex_y = get_max_activation_patch(ex_f_map, ex_img.shape[:2], patch_size)
                ax_ex.add_patch(
                    patches.Rectangle((ex_x, ex_y), patch_size, patch_size, linewidth=3, edgecolor='#00FF00',
                                      facecolor='none'))
            else:
                ax_ex.text(0.5, 0.5, "No\nData", ha='center')

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def run_prediction_flow(checkpoint_path=None, min_classes=1, num_prototypes=2):
    results_dir = CONFIG.get("results_dir", "./results")
    output_dir = os.path.join(results_dir, "epic_results")
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_path is None:
        filename = "epic_checkpoint.pth" if CONFIG.get("apply_epic", False) else "deeplab_checkpoint.pth"
        checkpoint_path = os.path.join(results_dir, filename)

    model = load_trained_model(checkpoint_path)

    dm = VOCDatasetManager(CONFIG)
    train_loader, test_loader = dm.get_dataloaders()
    dataset = test_loader.dataset

    db_path = os.path.join(output_dir, "exemplar_scores.pt")

    if os.path.exists(db_path):
        saved_data = torch.load(db_path, map_location="cpu")
        if isinstance(saved_data, dict) and 'scores' in saved_data:
            scores_matrix = saved_data['scores']
        else:
            scores_matrix = saved_data
    else:
        scores_matrix = build_exemplar_database(model, train_loader.dataset, db_path)

    dataset_size = len(dataset)

    if hasattr(dataset, 'images'):
        image_names = [os.path.splitext(os.path.basename(path))[0] for path in dataset.images]
    else:
        image_names = [str(i) for i in range(dataset_size)]

    folder_counts = {}
    pbar = tqdm(total=dataset_size, desc="Processing full test set")

    for idx in range(dataset_size):
        img_tensor, gt_mask = dataset[idx]
        img_id = image_names[idx]

        input_batch = img_tensor.unsqueeze(0).to(CONFIG["device"])

        with torch.no_grad():
            output = model(input_batch)['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            epic_features = model.classifier[4].last_disentangled_features

        unique_classes = np.unique(pred_mask)
        unique_classes = [int(c) for c in unique_classes if c != 0 and c != 255]

        num_classes_found = len(unique_classes)

        if num_classes_found < min_classes:
            pbar.update(1)
            continue

        B, C, H_feat, W_feat = epic_features.shape

        pred_mask_tensor = torch.tensor(pred_mask, device=CONFIG["device"])
        pred_mask_resized = F.interpolate(
            pred_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(H_feat, W_feat),
            mode='nearest'
        ).squeeze().long()

        chosen_channels = []
        chosen_scores = []
        chosen_classes = []

        features_2d = epic_features.squeeze(0).reshape(C, -1)
        probs = F.softmax(features_2d, dim=-1)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)

        for cls_id in unique_classes:
            cls_mask = (pred_mask_resized == cls_id)
            features_for_cls = epic_features[0, :, cls_mask]

            if features_for_cls.shape[1] == 0:
                continue

            max_acts_for_cls = features_for_cls.abs().max(dim=1)[0]

            combined_score_for_cls = max_acts_for_cls / (entropy + 1e-4)

            topk_scores, topk_indices = torch.topk(combined_score_for_cls, min(num_prototypes, C))

            for score, c_idx in zip(topk_scores, topk_indices):
                chosen_channels.append(c_idx.item())
                chosen_scores.append(score.item())
                chosen_classes.append(cls_id)

        if len(chosen_channels) == 0:
            pbar.update(1)
            continue

        unique_top_channels = np.unique(chosen_channels)
        exemplars_data = find_train_exemplars(
            model, train_loader.dataset, scores_matrix, unique_top_channels
        )

        class_dir = os.path.join(output_dir, f"class {num_classes_found}")
        os.makedirs(class_dir, exist_ok=True)

        save_filename = os.path.join(class_dir, f"{img_id}.png")

        visualize_full_epic_analysis(img_tensor, gt_mask, pred_mask, epic_features, exemplars_data, chosen_channels,
                                     chosen_scores, chosen_classes, save_path=save_filename)

        folder_counts[num_classes_found] = folder_counts.get(num_classes_found, 0) + 1
        pbar.update(1)

    pbar.close()

    print("\n[INFO] Generation completed. Saved objects status:")
    for num_cls, count in sorted(folder_counts.items()):
        print(f" -> Folder 'class {num_cls}': {count} images.")


if __name__ == "__main__":
    run_prediction_flow(min_classes=1, num_prototypes=2)