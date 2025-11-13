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
