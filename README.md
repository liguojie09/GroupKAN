# 🧠 GroupKAN: Rethinking Nonlinearity with Grouped Spline-based KAN Modeling for Efficient Medical Image Segmentation

:pushpin: Official PyTorch implementation of **GroupKAN: Rethinking Nonlinearity with Grouped Spline-based KAN Modeling for Efficient Medical Image Segmentation**

[[`Paper (arXiv)`](https://arxiv.org/abs/xxxx.xxxxx)] [[`Model Zoo`](https://github.com/liguojie09/GroupKAN/tree/main/checkpoints)]

---

## 🌟 Overview

**GroupKAN** revisits the design of nonlinear representations in medical image segmentation by introducing **grouped spline-based functional modeling**.  
It integrates two novel components — **Grouped KAN Transform (GKT)** and **Grouped KAN Activation (GKA)** — into a lightweight and interpretable U-shaped backbone.

<p align="center">
  <img src="https://github.com/liguojie09/GroupKAN/blob/main/docs/framework.png" alt="GroupKAN Framework" width="90%">
</p>
<p align="center"><b>Figure 1.</b> Overall architecture of GroupKAN. The Grouped KAN Transform and Activation efficiently model intra-group dependencies with learnable spline functions.</p>

---

## 💡 Key Highlights

- 🔹 **Efficient Grouped Nonlinearity:**  
  Splits features into groups and applies spline-based transformations, reducing full-channel complexity from O(C²) to **O(C²/G)**.

- 🔹 **Interpretable Functional Mapping:**  
  Each group learns localized nonlinearities, providing clearer feature attribution in medical segmentation.

- 🔹 **Compact Yet Powerful:**  
  Achieves **+1.1% IoU gain** over U-KAN with **47.6% fewer parameters (3.02M vs. 6.35M)**.

<p align="center">
  <img src="https://github.com/liguojie09/GroupKAN/blob/main/docs/efficiency_tradeoff.png" alt="Efficiency tradeoff" width="47%">
</p>
<p align="center"><b>Figure 2.</b> Accuracy–complexity comparison across models. GroupKAN achieves better accuracy with nearly half the parameters of U-KAN.</p>

---

## 📈 Quantitative Results

| Model | Params (M) | GFLOPs | Avg. IoU ↑ | Avg. F1 ↑ |
|:------|:-----------:|:-------:|:-----------:|:-----------:|
| U-Net | 31.04 | 436.9 | 75.84 | 85.37 |
| U-KAN | 6.35 | 14.02 | 78.69 | 87.26 |
| **GroupKAN (Ours)** | **3.02** | **7.72** | **79.80** | **88.07** |

---

## 🩻 Qualitative Results

<p align="center">
  <img src="https://github.com/liguojie09/GroupKAN/blob/main/docs/qualitative_results.png" alt="Segmentation Examples" width="99%">
</p>
<p align="center"><b>Figure 3.</b> GroupKAN produces more accurate and sharper segmentation results across BUSI, GlaS, and CVC datasets.</p>

---

## 🧠 Explainability

GroupKAN exhibits improved activation–mask alignment and interpretable feature distributions compared with U-KAN.

<p align="center">
  <img src="https://github.com/liguojie09/GroupKAN/blob/main/docs/explainability.png" alt="Explainability visualization" width="50%">
</p>
<p align="center"><b>Figure 4.</b> Activation maps align more closely with anatomical boundaries, showing improved interpretability.</p>

---

## ⚙️ Installation

```bash
git clone https://github.com/GroupKAN/GroupKAN.git
cd GroupKAN
conda create -n groupkan python=3.10
conda activate groupkan
pip install -r requirements.txt
```

## 🚀 Training & Evaluation

### 🧩 Training Example
```bash
python train.py --dataset busi --input_size 256 --epochs 400 --batch_size 8
```
### 🧪 Evaluation Example
```bash
python val.py --weights checkpoints/groupkan_best.pth --dataset glas
```

## 📦 Model Zoo

| Variant | Params (M) | IoU (%) | Checkpoint |
|----------|-------------|---------|-------------|
| GroupKAN-BUSI | 3.02 | 79.8 | [KANet_Busi](checkpoints/KANet_Busi/) |
| GroupKAN-CVC | 3.02 | 85.6 | [KANet_CVC](checkpoints/KANet_CVC/) |
| GroupKAN-GLAS | 3.02 | 87.5 | [KANet_GLAS](checkpoints/KANet_GLAS/) |

---

## 🛒 TODO

- [x] Release segmentation code  
- [x] Release pre-trained checkpoints  
- [ ] Add demo and visualization scripts  

---

## 🎈 Acknowledgements

This work is inspired by [U-KAN](https://arxiv.org/abs/2406.02918)  
and the Kolmogorov–Arnold Network (KAN).  
We thank the open-source community for their valuable contributions.

---

## 📬 Contact

For questions and collaborations, please reach out to  
**Guojie Li** (<liguojie@liverpool.ac.uk>).

