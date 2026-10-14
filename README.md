# Periocular Region-Based Deepfake Detection with Explainable AI

Detects AI-generated faces from the **periocular (eye) region only**, for situations where the full face is not available (masks, veils, cropped views). Two ImageNet-pretrained backbones, **Xception** and **EfficientNetB0**, are trained on aligned 2:1 eye-region crops and explained with **Grad-CAM**.

**Author:** Afrida Jahan Marjia, Department of CSE, Asian University of Bangladesh

## Pipeline

1. **ROI extraction** (`roi_extraction.ipynb`): MediaPipe landmarks, eye alignment, padding (0.35 x inner-eye distance), fixed 2:1 crop, Lanczos resize (112x224 for EfficientNetB0, 160x320 for Xception). A perceptual-hash (pHash) manifest is written for the leakage check.
2. **Training and evaluation** (`train_evaluate.ipynb`): leakage check and cleanup (training split only), two-phase training (frozen backbone, then fine-tune the top 20%), focal loss, threshold chosen on validation (Youden's J), test metrics, confusion matrix, ROC, Grad-CAM.

## Repository contents

| File                   | Purpose                                                                        |
| ---------------------- | ------------------------------------------------------------------------------ |
| `roi_extraction.ipynb` | Builds the periocular crops, manifest and failure log from the raw dataset zip |
| `train_evaluate.ipynb` | Trains, evaluates and explains a model; also runs the ablations                |
| `requirements.txt`     | Python packages (Colab already provides most of them)                          |

## Dataset

[Real vs. AI-Generated Faces (Kaggle)](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset). The data is **not** included in this repository. Download it, zip it with the structure `train/`, `test/`, `validate/` (each containing `0/` for real and `1/` for fake), and upload the zip to Google Drive.

## How to run (Google Colab, T4/L4 GPU)

Both notebooks read and write Google Drive paths. Edit the paths in the config cells if your folders differ.

**Step 1 - ROI extraction.** Open `roi_extraction.ipynb`, set `MODEL_NAME` and `DATASET_ZIP`, then run all cells. Run it once for each model.

| `MODEL_NAME`     | Crop size | Output on Drive                                                   |
| ---------------- | --------- | ----------------------------------------------------------------- |
| `efficientnetb0` | 112 x 224 | `roi_FDataset_rect_112x224.zip`, `_manifest.csv`, `_failures.csv` |
| `xception`       | 160 x 320 | `roi_FDataset_rect_160x320.zip`, `_manifest.csv`, `_failures.csv` |

**Step 2 - Training.** Open `train_evaluate.ipynb`, set the options below, choose a **T4/L4 GPU** runtime, and run all cells. Repeat for each seed.

| Option            | Values                           | Meaning                                                      |
| ----------------- | -------------------------------- | ------------------------------------------------------------ |
| `MODEL_NAME`      | `"efficientnetb0"`, `"xception"` | Backbone and input size                                      |
| `SEED`            | `42`, `123`, `2024`              | One run per seed                                             |
| `LOSS_TYPE`       | `"focal"` (default), `"bce"`     | Loss function ablation                                       |
| `LEAKAGE_CLEANUP` | `True` (default), `False`        | Leakage ablation: keep leaked duplicates in the training set |

If Colab disconnects, run the notebook again with the same settings and it resumes from the saved checkpoint.

Each run saves a metrics JSON (`<model>_seed<seed>_metrics.json`), the best checkpoints and the final `.keras` model to Drive.

## Results

Test set, 24,176 images. Mean over three seeds (42, 123, 2024).

| Model          | Input     | Accuracy | AUC    |
| -------------- | --------- | -------- | ------ |
| EfficientNetB0 | 112 x 224 | 93.73%   | 0.9827 |
| Xception       | 160 x 320 | 96.50%   | 0.9931 |

Periocular versus full-face input (seed 42):

| Model          | Full-face accuracy / AUC | Periocular accuracy / AUC |
| -------------- | ------------------------ | ------------------------- |
| Xception       | 96.89% / 0.9939          | 96.34% / 0.9926           |
| EfficientNetB0 | 94.18% / 0.9833          | 93.88% / 0.9831           |

Ablations (EfficientNetB0, seed 42): leakage-cleaned 93.88% versus raw 93.85%; focal loss 93.88% versus plain BCE 93.99%.

## Notes and limitations

- The leakage check detects **exactly equal** perceptual hashes only; it does not detect identity-level or source-image overlap.
- Xception (160x320) and EfficientNetB0 (112x224) use different input sizes, so their comparison also reflects resolution.
- The full-face baseline and the ablations are single-seed runs.
- If the manifest file is missing, the training notebook prints a warning and skips the leakage check.
