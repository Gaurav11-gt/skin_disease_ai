"""Grad-CAM (Selvaraju et al., 2017) for explainability."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) tensor. Returns (cam HxW in [0,1], logits, class_idx)."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach(), idx


def overlay(rgb_uint8, cam, alpha=0.4):
    h, w = rgb_uint8.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb_uint8).astype(np.uint8)
