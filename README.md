# SAMix

Official code for:  
**"SAMix: Calibrated and Accurate Continual Learning via Sphere-Adaptive Mixup and Neural Collapse"**

---

## 🛠️ Prerequisites

- **Python 3.8**  
  Check installation:
  ```bash
  python3.8 --version
  ```
- **pip for Python 3.8**  
  Check installation:
  ```bash
  python3.8 -m pip --version
  ```
- **wandb account**  
  - This project uses **wandb** for convenient logging.  
  - Create an account and log in before proceeding

     ```bash
         --wandb \
         --wandb_project_name your_project_name \
         --wandb_entity your_entity \
    ```

---

## 📦 Environment Setup

1. **Install virtualenv:**
    ```bash
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install virtualenv
    ```

2. **Create and activate virtual environment:**
    ```bash
    # Create new environment
    python3.8 -m virtualenv venv

    # Activate environment
    source venv/bin/activate  # On macOS/Linux
    # or
    .\venv\Scripts\activate  # On Windows
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install library for downloading the Tiny-ImageNet dataset:**
    ```bash
    pip uninstall googledrivedownloader
    pip install assets/googledrivedownloader-0.4-py2.py3-none-any.whl
    ```

    **Note**: Due to the large size of the Tiny-ImageNet dataset, please download this dataset manually, upload it to your Google Drive, and update the script src/datasets/tiny_imagenet_dataset.py accordingly ("file_id='id_file_name'") to enable downloading and usage.

> **Note**: To deactivate the virtual environment when you're done:
> ```bash
> deactivate
> ```

---

## ⚙️ Configuration

- **Set output folder:**  
  Edit `config/base.yaml` and update `root_save_folder` to your desired path.

- **wandb configuration:**  
  In each bash script, set `wandb_project_name` and `wandb_entity` as needed.

---

## 🚀 Running Experiments

### 4.1. CIFAR-10

- **FC-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/fc_nccl/fc_nccl_samix_cf10.sh
    ```
- **TA-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/ta_nccl/ta_nccl_samix_cf10.sh
    ```

### 4.2. CIFAR-100

- **FC-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/fc_nccl/fc_nccl_samix_cf100.sh
    ```
- **TA-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/ta_nccl/ta_nccl_samix_cf100.sh
    ```

### 4.3. Tiny-ImageNet

- **FC-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/fc_nccl/fc_nccl_samix_tiny_imagenet.sh
    ```
- **TA-NCCL with SAMix**
    ```bash
    MEM_SIZE=200 LOG_FOLDER_ZERO_MEM=not-use bash scripts/train/ta_nccl/ta_nccl_samix_tiny_imagenet.sh
    ```

> **Note 1:**  
> Set `LOG_FOLDER_ZERO_MEM` to the log folder containing 200 auxiliary samples if `MEM_SIZE=0` (memory-free setting).

> **Note 2:**  
> **To run without SAMix:**
> 1. Remove these arguments:
>     ```bash
>     --samix \
>     --loss_normal_samples fnc2 \
>     --weight_samix_loss 5.0 \
>     ```
> 2. Change `--loss nc_samix` to:
>     ```bash
>     --loss fnc2
>     ```
>     or
>     ```bash
>     --loss dr
>     ```
> 3. If using `--loss fnc2`, add `--focal_gamma` and `--main_mark`

---

## 🧪 Evaluation

**Modify:** CHECKPOINT_FOLDER & LOG_FOLDER, and then run:

### 5.1. CIFAR-10
```bash
CHECKPOINT_FOLDER=/path/check/point/folder LOG_FOLDER=/path/log/folder/ bash scripts/evaluate/eval_task_cf10.sh
```

### 5.2. CIFAR-100
```bash
CHECKPOINT_FOLDER=/path/check/point/folder LOG_FOLDER=/path/log/folder/ bash scripts/evaluate/eval_task_cf100.sh
```

### 5.3. Tiny-ImageNet
```bash
CHECKPOINT_FOLDER=/path/check/point/folder LOG_FOLDER=/path/log/folder/ bash scripts/evaluate/eval_task_tiny_imagenet.sh
```

---

## 📊 Network Calibration

**Modify bash file:** "scripts/calibration/run_calibration.sh", and then run:

```bash
    bash scripts/calibration/run_calibration.sh
```

This will compute and display:
- Expected Calibration Error (ECE) each task
- Overconfidence Error (OE) each task
- Average Expected Calibration Error (AECE) 
- Average Overconfidence Error (AOE)

---

## 📂 Project Structure

```
.
├── assets/                             # Library for downloading Tiny-ImageNet dataset
├── config/                             # Configuration files (e.g., base.yaml for experiment settings)
├── scripts/                            # Training, evaluation, and calibration bash scripts
├── src/                                # Main source code for SAMix and all modules
│   ├── args/                           # Argument parsing and configuration utilities
│   ├── backbones/                      # Backbone network architectures
│   ├── buffers/                        # Buffer management for continual learning
│   ├── datasets/                       # Dataset-related utilities and loaders
│   ├── distillation/                   # Stability losses
│   ├── plas_losses/                    # Plasticity losses
│   ├── utils/                          # General utility functions and helper modules
├── README.md                           # Project documentation
├── main_compute_calibration.py         # Main file for computing calibration
├── main_eval.py                        # Eval run file
├── main_train.py                       # Train run file
└── requirements.txt                    # Python dependencies
```


## 📁 Output Folder Structure

```
root_path: in 'root_save_folder' in config/base.yaml
├── data
├── experiments
│   └── job_run_name-yyyy_mm_dd_HH_MM_SS
│       ├── models
│       │   ├── encoder_cosine[_warm]
│       │   │   ├── task_{i}.pth
│       │   │   ├── args_task_{i}.json
│       │   ├── eval_linear_yyyy_mm_dd_HH_MM_SS
│       │   │   │── accuracy_task_{i}.txt
│       │   │   │── eval_args_task_{i}.json       
│       │   │   │── linear_model_eval_linear_yyyy_mm_dd_HH_MM_SS_task_{i}.pth
│       │────logs
│       │   │── encoder_cosine[_warm]
│       │   │   │── prototypes_dim={d}_k={n_cls}_seed={seed}.npy
│       │   │   │── subset_indices_random_{i}.npy
│       │   │   │── replay_indices_random_{i}.npy

