import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

class EPICDisentanglementConv2d(nn.Module):
    def __init__(self, original_conv):
        super().__init__()
        assert original_conv.kernel_size == (1, 1) or original_conv.kernel_size == 1

        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels

        self.M = nn.Parameter(torch.eye(self.in_channels))

        self.original_weight = nn.Parameter(original_conv.weight.clone(), requires_grad=False)
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.last_disentangled_features = None

    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 1)
        disentangled_features = torch.matmul(x_perm, self.M)
        self.last_disentangled_features = disentangled_features.permute(0, 3, 1, 2)

        M_inv = torch.linalg.inv(self.M)
        M_inv_T = M_inv.t()

        w_flat = self.original_weight.view(self.out_channels, self.in_channels)
        w_new_flat = torch.matmul(w_flat, M_inv_T)
        w_new = w_new_flat.view(self.out_channels, self.in_channels, 1, 1)

        return F.conv2d(self.last_disentangled_features, w_new, self.bias)

def get_deeplab_model(num_classes, apply_epic=False):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)

    current_out_channels = model.classifier[4].out_channels

    if num_classes != current_out_channels:
        print(f"Num classes mismatch ({current_out_channels} vs {num_classes}). Replacing classifier...")

        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        if model.aux_classifier:
            in_channels_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
    else:
        print(f"Keeping pretrained weights for {num_classes} classes.")

    if apply_epic:
        model.classifier[4] = EPICDisentanglementConv2d(model.classifier[4])

        for name, param in model.named_parameters():
            if "classifier.4.M" not in name:
                param.requires_grad = False
        print("All layers frozen except EPIC Matrix M.")

    return model