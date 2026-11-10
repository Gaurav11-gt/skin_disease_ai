# Research-paper guide

> No results are included here. Run the experiments yourself and report exactly what you get - never copy numbers from other papers as yours.

## 1. Pick a clear contribution (reviewers need ONE sentence)
Suggested title: *"A Hybrid EfficientNet–CNN Attention Network with Calibrated Confidence and Grad-CAM Explanations for Skin Lesion Classification"*
Claimed contributions (only claim what your experiments show):
1. Hybrid architecture: pretrained backbone + SE-attention CNN head with avg‖max pooling.
2. Leakage-free lesion-level evaluation protocol (many published HAM10000 results are inflated by image-level splits).
3. Calibrated confidence (temperature scaling, ECE) + Grad-CAM + safety-oriented recommendation rule.

## 2. Experiments checklist (all scripted)
- [ ] Baselines: simple_cnn, resnet50, effnet_b0 vs hybrid_resnet50, hybrid_effnet; 3 seeds, report mean ± std
- [ ] Metrics: accuracy, **balanced accuracy, macro-F1, macro-AUC, melanoma recall** (accuracy alone is misleading: ~67% of data is "nv"), ECE
- [ ] Ablations: no class weights / no 2-stage fine-tuning / no preprocessing / no pretraining / (extra: remove SE block, remove max-pool)
- [ ] Significance: McNemar test or paired bootstrap CI between best baseline and proposed model (use test_probs.npy)
- [ ] Calibration: reliability diagram + ECE before/after temperature scaling
- [ ] Explainability: Grad-CAM grid of correct and **incorrect** cases; discuss failure modes honestly
- [ ] External validation (strongly boosts acceptance): train on HAM10000, test on a different set (e.g. ISIC 2019 images not in HAM10000, PH2) and report the drop
- [ ] Per-class error analysis: mel↔nv, akiec↔bkl confusion

## 3. Paper structure (IEEE/Springer style, ~8 pages)
1. **Abstract** (problem, method, 3 key numbers, limitation)  2. **Introduction** (motivation, gap, contributions)
3. **Related work** (CNNs for dermatology, transfer learning, attention, calibration/XAI)
4. **Dataset & protocol** (HAM10000, lesion-level split, class imbalance table)
5. **Method** (preprocessing, architecture figure, two-stage training, loss, calibration, Grad-CAM, recommendation logic)
6. **Experiments** (setup/hyperparameters, main table, ablation table, calibration, XAI, external test)
7. **Discussion** (clinical relevance, bias/skin-tone, failure cases, limitations)  8. **Conclusion & future work**
Figures: pipeline, architecture, confusion matrix, ROC, reliability diagram, Grad-CAM grid, training curves.

## 4. Core references (verify details before citing)
- Tschandl, Rosendahl, Kittler. *The HAM10000 dataset…* Scientific Data, 2018.
- Tan & Le. *EfficientNet…* ICML 2019.  - He et al. *Deep Residual Learning…* CVPR 2016.
- Hu, Shen, Sun. *Squeeze-and-Excitation Networks.* CVPR 2018.  - Selvaraju et al. *Grad-CAM.* ICCV 2017.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
- Esteva et al. *Dermatologist-level classification of skin cancer with deep neural networks.* Nature 2017.
- Codella et al. *Skin Lesion Analysis Toward Melanoma Detection (ISIC 2017/2018).*  - Combalia et al. *BCN20000.* 2019.

## 5. Where to submit
Student-friendly: IEEE/Springer conferences (ICCCNT, ICCCS, ICICV, ICACDS), journals like *Biomedical Signal Processing and Control*,
*Computers in Biology and Medicine*, *IEEE Access*, *Multimedia Tools and Applications*, *Scientific Reports*. Check indexing/predatory status
(avoid pay-to-publish "journals"); ask your guide for the target venue.

## 6. Pitfalls reviewers catch
- Image-level split leakage (we avoid it). - Tuning on the test set (we select on val only).
- Reporting only accuracy. - No external validation. - Claiming "diagnosis" - say "screening support".
- Ignoring skin-tone bias and dermoscopy-vs-phone domain gap. - Not releasing code (put this repo on GitHub).

## 7. Suggested 12-week plan
W1-2 literature + data/splits · W3 baselines · W4-5 hybrid model · W6 ablations · W7 calibration + Grad-CAM · W8 external test
W9 app/demo · W10-11 write paper + report · W12 polish, plagiarism check, submit.
