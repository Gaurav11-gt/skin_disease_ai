import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import CLASSES
from dataset import get_transforms
from gradcam import GradCAM, overlay
from knowledge import explain
from models import build_model
from preprocess import preprocess_image
from PIL import Image


def blur_score(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


class Predictor:
    def __init__(self, run_dir, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run = Path(run_dir)
        ck = torch.load(run / "best.pt", map_location=self.dev)
        self.args = ck["args"]
        self.model = build_model(self.args["model"], len(CLASSES), pretrained=False).to(self.dev)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.T = json.load(open(run / "temperature.json"))["temperature"] if (run / "temperature.json").exists() else 1.0
        self.tf = get_transforms(self.args["img_size"], train=False)
        self.cam = GradCAM(self.model, self.model.cam_layer)

    def predict(self, pil_img):
        rgb = np.array(pil_img.convert("RGB"))
        quality_warning = blur_score(rgb) < 30
        if self.args.get("preprocessed"):
            bgr = preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 256)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.dev)
        cam, logits, idx = self.cam(x)
        probs = F.softmax(logits.cpu() / self.T, 1)[0].numpy()      # calibrated
        code = CLASSES[int(probs.argmax())]
        if int(probs.argmax()) != idx:                                # (rare) temp scaling never changes argmax
            cam, _, _ = self.cam(x, int(probs.argmax()))
        pdict = {c: float(p) for c, p in zip(CLASSES, probs)}
        text, step, risk = explain(code, float(probs.max()), pdict)
        view = cv2.resize(rgb, (self.args["img_size"],) * 2)
        return dict(code=code, confidence=float(probs.max()), probs=pdict, explanation=text, next_step=step,
                    risk=risk, heatmap=overlay(view, cam), quality_warning=quality_warning)
