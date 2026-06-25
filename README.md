# Dual_SEnet

Dual_SEnet is a deep learning project for simultaneous prediction of distant metastasis (DM) and local recurrence (LR) in HNC patients. The model takes CT images, three-dimensional dose distributions, and GTV masks as input, and outputs risk scores for both DM and LR.

The project covers the full analysis workflow, including data preprocessing, dual-task deep learning model training and inference, Grad-CAM/AVH analysis, and statistical evaluation and visualization in R.

## Project Structure

```text
Dual_SEnet/
├── data_processing/      # DICOM data organization, checking, NIfTI conversion, and resampling
├── Deeplearning/         # Dual-task SE-ResNet model, training, testing, Grad-CAM, and AVH calculation
├── AVH_analysis/         # AVH curve plotting, group-wise statistical tests, and p-value visualization
├── Rcode/                # C-index, timeROC, DCA, KM curves, bootstrap, and other evaluation analyses
├── Radiomics/            # Radiomics mask generation and CT/dose feature extraction
└── README.md
```

## Method Overview

The main model is implemented in `Deeplearning/models/dual_task_model.py` and uses a 3D SE-ResNet architecture. The network contains two input branches:

- `CT + GTV mask`
- `Dose + GTV mask`

The two branches are processed by initial convolutional layers and then fused. The model shares low-level features and splits into two high-level task-specific branches for DM and LR. Each branch outputs a risk score for survival analysis and event prediction.

During training, a weighted Cox/DeepSurv-style survival loss is computed separately for DM and LR and optimized jointly:

```text
loss = loss_DM + loss_LR
```

Evaluation metrics include AUC, specificity, sensitivity, C-index, and 1/3/5-year time-dependent AUC.

## Data Requirements

The deep learning module expects one folder per patient. Each patient folder should contain:

```text
patient_id/
├── CT.nii.gz
├── DOSE.nii.gz
└── GTV_mask.nii.gz
```

The follow-up outcome file should be an Excel file. The current code reads data from different sheets through `read_follow_data()`. Each sheet uses the following columns by default:

| Column index | Meaning |
| --- | --- |
| 0 | Patient ID |
| 1 | DM event label |
| 2 | DM event time |
| 4 | LR event label |
| 5 | LR event time |

The cohort sheet names used in the current training and testing scripts include `HGJ`, `CHUS`, `CHUM`, and `HMR`. By default, the training script uses `HGJ + CHUS` as the training set and `CHUM` as the internal validation set. The testing script additionally uses `HMR` as the external validation set.

## Workflow

### 1. Data Preprocessing

`s2_Dcm2nii.py` reads CT, RS, and RD files (DICOM files), then generates and resamples:

```text
CT.nii.gz
DOSE.nii.gz
GTV_mask.nii.gz
```

The default resampling spacing is `3.0 x 3.0 x 3.0 mm`. Before running the scripts, update local paths such as `base_data_dir` and `save_dir` according to your data location.

### 2. Model Training

The training entry point is:

```bash
python Deeplearning/main_train.py
```

Common arguments include:

```text
--image_dir     Directory of preprocessed NIfTI data
--flag_path     Follow-up outcome Excel file
--save_name     Experiment name for saving outputs
--batch_size    Batch size
--epochs        Number of training epochs
--lr            Initial learning rate
--resume        Whether to resume training
--resume_path   Directory of the checkpoint for resumed training
```

Note: `main_train.py` currently uses `parser.parse_args(args=[])`, which ignores command-line arguments. To enable command-line arguments, change it to:

```python
args = parser.parse_args()
```

### 3. Model Inference and Evaluation

The testing entry point is:

```bash
python Deeplearning/main_test.py
```

The script loads a specified model, for example:

```text
./experiments/Dual_task_CD/inter_best_cindex_model.pth
```

It outputs DM/LR risk scores, event labels, follow-up times, and evaluation metrics for the training set, internal validation set, and external test set. Results are saved as an Excel file:

```text
experiments/Dual_task_CD/result_<model_name>.xlsx
```

### 4. Grad-CAM and AVH Analysis

Grad-CAM scripts are located in `Deeplearning/`:

```bash
python Deeplearning/get_GradCAM.py
python Deeplearning/plot_Grad_CAMone.py
```

AVH analysis scripts are located in `AVH_analysis/`:

```bash
python Deeplearning/cal_AVH.py
python AVH_analysis/plot_one_AVH.py
python AVH_analysis/sta_test.py
python AVH_analysis/plot_pvalue.py
```

This part is mainly used to analyze the cumulative distribution of model attention maps under different GTV regions and dose thresholds, followed by statistical comparisons between high- and low-risk groups.

### 5. Statistical Analysis in R

The `Rcode/` directory contains scripts for model performance evaluation, bootstrap confidence intervals, KM curves, and DCA analysis:

```text
boost_results.R       # Bootstrap evaluation of C-index, time AUC, sensitivity, and specificity
get_plot_dl_and_rf.R  # DCA, timeROC, and bar plots for deep learning and radiomics models
km_plot_3y.R          # 3-year KM curves using Youden index
km_plot_25.R          # Top 25% KM analysis
Radiomics_Cox.R       # Radiomics Cox modeling
temp_add_clinical.R   # Clinical information processing/merging
```

These scripts currently contain local absolute paths. Before running them, update the data paths and output directories for your own environment.

## Additional: Radiomics Analysis

`s2_get_features.py` uses PyRadiomics to extract features from CT and dose images. The parameter files are:

```text
Radiomics/params_ct.yaml
Radiomics/params_dose.yaml
```

## Notes

- The code contains several local absolute paths. Update data paths, model paths, and output paths before reproducing the experiments.
- Input filenames must match the dataloader: `CT.nii.gz`, `DOSE.nii.gz`, and `GTV_mask.nii.gz`.
- CT images are clipped to `[-400, 400]` and normalized to `[0, 1]` by default.
- Dose images are normalized from `[0, 87]` to `[0, 1]` by default.
- Training uses crop/pad to `128 x 128 x 64` by default.
- `main_train.py` currently fixes the GPU to `0`; modify `CUDA_VISIBLE_DEVICES` if a different GPU is needed.

## Citation

If you use this project in your research, please state that the model is used for simultaneous prediction of DM and LR after radiotherapy in patients with head and neck cancer, with CT, three-dimensional dose distribution, and GTV mask as input modalities.
