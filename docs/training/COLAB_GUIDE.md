# Google Colab Training Guide for Stick-Gen

This guide is designed for beginners who have never used Google Colab. It will walk you through training your model using Google's free GPUs.

## What is Google Colab?
Think of it as a computer that runs in your web browser. Google lets you use their powerful GPUs (Graphics Processing Units) for free for up to ~12 hours at a time. This is perfect for training deep learning models like ours.

---

## Phase 1: Prepare Your Files (On Your Mac)

Before going to Colab, we need to package your code and data.

1.  **Generate the Data**:
    Run the conversion script on your Mac to create the training file.
    ```bash
    # Run this in your terminal
    python src/data_gen/convert_100style.py
    # This creates: data/100style_processed.pt (or similar)
    # Ensure you have the final merged dataset: data/train_data_final.pt
    ```

2.  **Zip Your Code**:
    We need to send your code to Colab. The easiest way is to zip the project folder, **excluding** the heavy `data` folder (since we'll upload the processed data separately).
    
    Run this command in your terminal (from the `stick-gen` root):
    ```bash
    # Create a zip file of your code, config, and requirements
    zip -r stick_gen_code.zip src configs scripts requirements.txt
    ```

3.  **You should now have two key files**:
    *   `stick_gen_code.zip` (Your code)
    *   `data/train_data_final.pt` (Your processed training data)

---

## Phase 2: Google Drive Setup

1.  Go to [Google Drive](https://drive.google.com).
2.  Create a new folder named **`StickGen_Training`**.
3.  **Upload** the two files from Phase 1 into this folder:
    *   `stick_gen_code.zip`
    *   `train_data_final.pt`

---

## Phase 3: Running in Colab

1.  Open [Google Colab](https://colab.research.google.com).
2.  Click **"New Notebook"**.
3.  **Enable GPU**:
    *   Go to the menu: **Runtime** > **Change runtime type**.
    *   Under "Hardware accelerator", select **T4 GPU**.
    *   Click **Save**.

4.  **Copy and Paste** the following code blocks into the notebook cells. Run them one by one (click the "Play" button next to the cell).

### Cell 1: Connect to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
*It will ask for permission to access your Drive. Click "Connect to Google Drive" and approve.*

### Cell 2: Setup Workspace
This copies your files from Drive to the Colab machine (which is faster than reading from Drive directly).
```python
import os

# Create project directory
os.makedirs('/content/stick-gen/data', exist_ok=True)

# Copy and unzip code
!cp "/content/drive/MyDrive/StickGen_Training/stick_gen_code.zip" /content/stick-gen/
!unzip -q /content/stick-gen/stick_gen_code.zip -d /content/stick-gen/

# Copy dataset
!cp "/content/drive/MyDrive/StickGen_Training/train_data_final.pt" /content/stick-gen/data/

print("✅ Workspace setup complete!")
```

### Cell 3: Install Dependencies
```python
%cd /content/stick-gen
!pip install -r requirements.txt
```

### Cell 4: Run Training
```python
# Run the training script
!python src/train/train.py --config configs/base.yaml
```

---

## Phase 4: Saving Results

When training finishes (or if you stop it early), you'll want to save the trained model back to your Google Drive so you don't lose it when the Colab session ends.

### Cell 5: Save Model to Drive
```python
import shutil
import datetime

# Create a timestamped folder in Drive
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"/content/drive/MyDrive/StickGen_Training/models_{timestamp}"
os.makedirs(save_path, exist_ok=True)

# Copy checkpoints
!cp *.pth "{save_path}/"

print(f"✅ Model saved to Google Drive: {save_path}")
```
