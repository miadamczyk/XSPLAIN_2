import os
import torch
import random
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
        for images, _ in tqdm(loader):
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

    fig = plt.figure(figsize=(18, 4 * (num_rows + 1)))
    gs = fig.add_gridspec(num_rows + 1, 5, hspace=0.4, wspace=0.1)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(img_vis)
    ax_input.set_title("Oryginalny obraz")

    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.imshow(gt_mask.cpu().numpy(), cmap='viridis')
    ax_gt.set_title("Ground Truth")

    ax_pred = fig.add_subplot(gs[0, 2])
    ax_pred.imshow(pred_mask, cmap='viridis')
    ax_pred.set_title("Predykcja")

    for row_idx, c_idx in enumerate(top_channels):
        grid_row = row_idx + 1
        cls_id = chosen_classes[row_idx]

        ax_test = fig.add_subplot(gs[grid_row, 0])
        ax_test.imshow(img_vis)

        if top_scores is not None:
            ax_test.set_title(f"Klasa: {cls_id} | Kanał: {c_idx}\nWynik EPIC: {top_scores[row_idx]:.4f}")

        abs_feature = torch.abs(features[0, c_idx]).cpu().numpy()
        tx, ty = get_max_activation_patch(abs_feature, img_size, patch_size)

        ax_test.add_patch(
            patches.Rectangle((tx, ty), patch_size, patch_size, linewidth=3, edgecolor='red', facecolor='none'))

        train_list = exemplars_data.get(c_idx, [])
        for col_idx in range(1, 5):
            ax_ex = fig.add_subplot(gs[grid_row, col_idx])
            if col_idx - 1 < len(train_list):
                ex_score, ex_tensor, ex_f_map = train_list[col_idx - 1]
                ex_img = get_img_for_vis(ex_tensor)
                ax_ex.imshow(ex_img)
                ax_ex.set_title(f"Score: {ex_score:.4f}")

                ex_x, ex_y = get_max_activation_patch(ex_f_map, ex_img.shape[:2], patch_size)
                ax_ex.add_patch(
                    patches.Rectangle((ex_x, ex_y), patch_size, patch_size, linewidth=3, edgecolor='#00FF00',
                                      facecolor='none'))
            else:
                ax_ex.text(0.5, 0.5, "Brak\ndanych", ha='center')

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# Zmiana tutaj: dodano argument num_prototypes
# Zastąp w swoim pliku CAŁĄ funkcję run_prediction_flow tym kodem:

def run_prediction_flow(checkpoint_path=None, num_images_to_process=5, min_classes=1, num_prototypes=1):
    results_dir = CONFIG.get("results_dir", "./results")
    output_dir = os.path.join(results_dir, "epic_results")
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_path is None:
        filename = "epic_checkpoint.pth" if CONFIG.get("apply_epic", False) else "deeplab_checkpoint.pth"
        checkpoint_path = os.path.join(results_dir, filename)

    model = load_trained_model(checkpoint_path)

    dm = VOCDatasetManager(CONFIG)
    train_loader, test_loader = dm.get_dataloaders()

    db_path = os.path.join(output_dir, "exemplar_scores.pt")
    if not os.path.exists(db_path):
        scores_matrix = build_exemplar_database(model, train_loader.dataset, db_path)
    else:
        scores_matrix = torch.load(db_path, map_location="cpu")

    dataset_size = len(test_loader.dataset)
    random_indices = random.sample(range(dataset_size), dataset_size)

    processed_count = 0
    pbar = tqdm(total=min(num_images_to_process, dataset_size),
                desc=f"Przetwarzanie (min. klas: {min_classes}, prototypy/klasę: {num_prototypes})")

    for idx in random_indices:
        if processed_count >= num_images_to_process:
            break

        img_tensor, gt_mask = test_loader.dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(CONFIG["device"])

        # 1. Najpierw robimy predykcję (musimy mieć predykcję, żeby sprawdzić, co wykrył model)
        with torch.no_grad():
            output = model(input_batch)['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            epic_features = model.classifier[4].last_disentangled_features

        # 2. Szukamy klas w masce predykcji (używamy numpy, bo pred_mask to tablica numpy)
        unique_classes = np.unique(pred_mask)
        unique_classes = [int(c) for c in unique_classes if c != 0 and c != 255]

        # 3. Jeśli model nie wykrył minimalnej liczby klas, pomijamy zdjęcie
        if len(unique_classes) < min_classes:
            continue

        features_2d = epic_features.squeeze(0).reshape(epic_features.shape[1], -1)

        probs = F.softmax(features_2d, dim=-1)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)

        channel_relevance = torch.abs(features_2d).max(dim=-1)[0]
        combined_score = channel_relevance / (entropy + 1e-4)

        sorted_indices = torch.argsort(combined_score, descending=True).cpu().numpy()
        sorted_scores = combined_score[sorted_indices].cpu().numpy()

        chosen_channels = []
        chosen_scores = []
        chosen_classes = []

        img_size = gt_mask.shape
        patch_size = 40

        # --- Analizujemy wszystkie kanały na podstawie PREDYKCJ ---
        channel_hits = []
        for i in range(len(sorted_indices)):
            c_idx = sorted_indices[i]
            score = sorted_scores[i]

            abs_feature = torch.abs(epic_features[0, c_idx]).cpu().numpy()
            tx, ty = get_max_activation_patch(abs_feature, img_size, patch_size)

            # Wycianamy patch z maski PREDYKCJ, nie z gt_mask!
            patch_mask = pred_mask[ty:ty + patch_size, tx:tx + patch_size]

            patch_classes = np.unique(patch_mask).tolist()
            channel_hits.append({
                'c_idx': c_idx,
                'score': score,
                'classes': patch_classes
            })

        # --- Wybieranie prototypów dla klas ---
        for cls in unique_classes:
            found_for_this_class = 0

            for hit in channel_hits:
                if cls in hit['classes']:
                    chosen_channels.append(hit['c_idx'])
                    chosen_scores.append(hit['score'])
                    chosen_classes.append(cls)
                    found_for_this_class += 1

                    if found_for_this_class >= num_prototypes:
                        break

        if len(chosen_channels) == 0:
            continue

        top_channels = np.array(chosen_channels, dtype=int)
        top_scores = np.array(chosen_scores, dtype=float)

        unique_top_channels = np.unique(top_channels)
        exemplars_data = find_train_exemplars(model, train_loader.dataset, scores_matrix, unique_top_channels)

        save_filename = os.path.join(output_dir, f"epic_analysis_{processed_count + 1}_img_{idx}.png")
        visualize_full_epic_analysis(img_tensor, gt_mask, pred_mask, epic_features, exemplars_data, top_channels,
                                     top_scores, chosen_classes, save_path=save_filename)

        processed_count += 1
        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    # Przykład użycia:
    # 5 obrazów | Min. 2 klasy na obrazie | 3 najlepsze kanały (prototypy) wyciągnięte na każdą klasę
    run_prediction_flow(num_images_to_process=5, min_classes=3, num_prototypes=3)