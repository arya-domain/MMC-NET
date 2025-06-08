# MMC-NET

Official repository for the MMC-NET paper.

---

## 📁 Dataset Structure

Ensure the dataset directory is organized as follows:

```
dataset/
├── ACDC
├── Synapse
└── Lung
```

* **ACDC & Synapse Dataset**: [Download from Kaggle](https://www.kaggle.com/datasets/aryandas2021/synapse-and-acdc)
* **Lung Dataset (Original)**: [Download from Google Drive](https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu)
* **Lung Dataset (Preprocessed)**: Will be provided upon paper acceptance.

---

## 📦 Pretrained Weights

Pretrained encoder model weights can be downloaded from:

👉 [Pretrained Weights - Google Drive](https://drive.google.com/drive/folders/1k-s75ZosvpRGZEWl9UEpc_mniK3nL2xq)

After downloading, create a folder named `pretrained_pth` in the project root and place the `.pth` files inside:

```
pretrained_pth/
├── maxxvit_rmlp_small_rw_256_sw-37e217ff.pth
└── maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth
```

---

## Training Instructions

### Train on ACDC Dataset

To start training on the ACDC dataset, run:

```bash
python train_ACDC.py
```

---

## 📌 Training Code Availability

| Dataset | Training Code      | Status             |
| ------- | ------------------ | ------------------ |
| ACDC    | `train_ACDC.py`    | ✅ Available        |
| Synapse | `train_Synapse.py` | 🔒 Upon Acceptance |
| Lung    | `train_Lung.py`    | 🔒 Upon Acceptance |

---


