## Preprocessing Summary

This preprocessing step prepares the dataset for pneumonia classification based on the **RSNA Pneumonia Detection Challenge**.

### Overview
- **Data Source:** RSNA Chest X-ray dataset (`stage_2_train_images` and `stage_2_train_labels.csv`)
- **Goal:** Prepare the data for a binary classification task, *Normal (0)* vs. *Pneumonia (1)*

### Steps Performed
1. **Load and clean labels**  
   - Reads the CSV file, removes duplicates, and uses the `Target` column as the class label.

2. **Read DICOM images**  
   - Each image is loaded using `pydicom.dcmread()`, then scaled to the 0–1 range.

3. **Resize and convert**  
   - Images are resized to **224×224** pixels with OpenCV (`cv2.resize`)  
   - Converted to `float16` for smaller file size

4. **Split the dataset**  
   - First **24,000** images → training set  
   - Remaining images → validation set  
   - Stored in folders:  
     - `Processed/train/0/` and `Processed/train/1/`  
     - `Processed/val/0/` and `Processed/val/1/`

5. **Compute dataset statistics**  
   - Calculates mean and standard deviation (from training set only)  
   - These values are used for normalization during model training

6. **Save preprocessed data**  
   - Each image saved as `.npy` array for efficient loading in TensorFlow or PyTorch

### Key Points
- Images are **grayscale** arrays with shape `(224, 224)`  
- Normalization is based on training set statistics  
- Folder names (`0`, `1`) determine class labels  
- The data is ready for direct use in a CNN classifier


## PyTorch Implementation Overview

This section describes the original pneumonia classification pipeline implemented in **PyTorch** and **PyTorch Lightning**.

### 1. Imports

The notebook uses:

- `torch`, `torchvision` for model building and training
- `torchvision.transforms` for data augmentation and normalization
- `torchmetrics` for evaluation metrics (accuracy, precision, recall, confusion matrix)
- `pytorch_lightning` for high-level training loops (`LightningModule`, `Trainer`)
- `tqdm`, `numpy`, `matplotlib` for progress bars, array ops, and plotting

---

### 2. Transforms

Data preprocessing and augmentation are defined using `torchvision.transforms`:

- **Training transforms**:
  - Convert `.npy` arrays to tensors
  - Normalize using dataset mean and std (computed in the preprocessing notebook)
  - Apply data augmentation such as:
    - `RandomAffine` (small rotations, translations, scaling)
    - `RandomResizedCrop` to random locations/scales

- **Validation transforms**:
  - Convert to tensor
  - Apply the same normalization (no augmentation)

---

### 3. Datasets

The dataset is loaded from preprocessed `.npy` files using:

- `torchvision.datasets.DatasetFolder`  
  - Root directories: `Processed/train/` and `Processed/val/`  
  - `loader=load_file`, where `load_file(path)` loads a `.npy` array and casts it to `float32`
  - `extensions="npy"` ensures only preprocessed NumPy images are used
- Class labels are inferred from folder names (e.g. `0` = normal, `1` = pneumonia)

---

### 4. DataLoaders

The datasets are wrapped in `DataLoader` objects:

- `train_loader`:
  - Shuffled each epoch
  - Batches of fixed size (e.g., 64)
  - Optional `num_workers` for parallel loading

- `val_loader`:
  - No shuffling
  - Same batch size
  - Used only for validation during training

---

### 5. Model

A custom `LightningModule` called `PneumoniaModel` is defined:

- Backbone: `torchvision.models.resnet18`
  - First convolution modified to accept **1-channel** grayscale input instead of 3-channel RGB
- Final layer:
  - Fully connected layer outputs a single **logit** for binary classification
- Loss:
  - `BCEWithLogitsLoss` (binary cross-entropy with logits)
  - Optional `pos_weight` to handle class imbalance
- Metrics:
  - Accuracy tracked using `torchmetrics` during training and validation steps

---

### 6. Trainer

Training is managed with `pytorch_lightning.Trainer`:

- Configured with:
  - GPU or CPU accelerator
  - Maximum number of epochs
  - Logging via `TensorBoardLogger`
  - Callbacks:
    - `ModelCheckpoint` to save the best model based on validation performance
    - (Optional) Early stopping

---

### 7. Fit

The training loop is started with:

```python
trainer.fit(model, train_loader, val_loader)

```
---
### 8. Evaluation

After training is complete:

- The best checkpoint is loaded back into `PneumoniaModel`.
- Predictions and ground-truth labels are collected over the validation set.
- Evaluation metrics are computed using `torchmetrics`, including:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **Confusion Matrix** at:
    - Threshold **0.5** (default)
    - Threshold **0.3** (for sensitivity analysis)

These evaluation results provide a clear view of how well the model distinguishes between normal and pneumonia chest X-ray images.

