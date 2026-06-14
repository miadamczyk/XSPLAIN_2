import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from config import CONFIG


def get_voc_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def visualize_prediction(model, loader, device, save_name="prediction.png"):
    model.eval()

    try:
        images, masks = next(iter(loader))
    except StopIteration:
        return

    cmap = get_voc_colormap()

    with torch.no_grad():
        output = model(images.to(device))['out']
        preds = torch.argmax(output, dim=1).cpu().numpy()

    gt_img = masks[0].numpy()
    pred_img = preds[0]

    gt_img[gt_img == 255] = 0

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(cmap[gt_img])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cmap[pred_img])
    plt.axis('off')

    plt.tight_layout()

    results_dir = CONFIG.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_name)
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(model, filename="deeplab_checkpoint.pth"):
    results_dir = CONFIG.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    filename_base = os.path.basename(filename)
    save_path = os.path.join(results_dir, filename_base)
    torch.save(model.state_dict(), save_path)