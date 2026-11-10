# AI-Assisted Skin Disease Classification (Hybrid CNN + Transfer Learning)

Pipeline: Skin image → preprocessing (hair removal, colour constancy) → EfficientNet/ResNet backbone → extra CNN head
(2 conv blocks + Squeeze-Excitation + avg‖max pooling) → 7-class prediction → calibrated confidence → Grad-CAM
explanation → recommended next step.  **Research prototype – supports, never replaces, a dermatologist.**

## 1. Setup
```bash
pip install -r requirements.txt        # needs a GPU for sensible training time (free Colab T4 is enough)
```
## 2. Data (HAM10000, free)
Download from Kaggle ("Skin Cancer MNIST: HAM10000") or Harvard Dataverse (Tschandl et al., 2018) and unzip into
`data/HAM10000/` (must contain `HAM10000_metadata.csv` + the jpgs).
```bash
python data_prep.py --raw_dir data/HAM10000 --out_dir data/splits        # lesion-level split, no leakage
python preprocess.py --splits_dir data/splits --out_dir data/processed   # hair removal + colour constancy (once)
```
## 3. Train & evaluate one model
```bash
python train.py --model hybrid_effnet --data_dir data/processed --preprocessed --epochs 30 --seed 42
# -> outputs/hybrid_effnet_s42/{best.pt, metrics.json, confusion_matrix.png, roc.png, classification_report.txt, ...}
```
Models: `hybrid_effnet` (proposed), `hybrid_resnet50`, `effnet_b0`, `resnet50`, `simple_cnn`.

## 4. All experiments for the paper (5 models × 3 seeds + 4 ablations)
```bash
./run_experiments.sh        # then results_table.md / results_table.tex are generated
```
## 5. Demo app
```bash
streamlit run app.py -- --run outputs/hybrid_effnet_s42
```
## Files
| file | purpose |
|---|---|
| data_prep.py | lesion-grouped stratified split 80/10/10 |
| preprocess.py | hair removal, Shades-of-Gray, resize |
| dataset.py | loaders + augmentation |
| models.py | proposed hybrid + baselines |
| train.py | 2-stage transfer learning, class weights, label smoothing, AMP, early stopping |
| evaluate.py | metrics, ROC, confusion matrix, ECE + temperature scaling |
| gradcam.py / knowledge.py / inference.py / app.py | explainability, advice text, demo |
| aggregate.py | mean±std tables (Markdown + LaTeX) |

## Colab quick start
Runtime → GPU; `!git clone <your repo>`; `%cd skin-disease-ai`; `!pip install -r requirements.txt`; mount Drive for the dataset; run steps 2–3.

## Known limits (state these in the paper too)
HAM10000 = dermoscopic images, mostly lighter skin types, one-site-heavy; phone photos and darker skin tones are out-of-distribution.
