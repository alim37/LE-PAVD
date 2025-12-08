## 🐍 Python Virtual Environment Setup

This project uses Python virtual environments to ensure consistent dependencies and reproducible results.

### Install `python-venv`

On **Ubuntu / Debian-based systems**:
```bash
sudo apt update
sudo apt install -y python3-venv
```


Verify installation:
```
python3 -m venv --help
```
Create a Virtual Environment

From the project root directory:
```
python3 -m venv venv
```
Activate the Virtual Environment
```
source venv/bin/activate
```
You should see (venv) in your terminal prompt.

Install Project Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```


## Model Training

To run an individual training experiment, use:

```
cd /model/
python3 train.py {path to cfg} {path to dataset} {name of experiment} {dataset split}
```


## Model Evaluation

To evaluate an individual model, use:

```
cd /model/
python3 evaluate.py {path to cfg} {path to dataset} {path to model weights}
```

## ✅ Tested On

### Hardware
<!-- - **CPU:** AMD Ryzen 9 7900X (24 threads @ up to 5.73 GHz) -->
- **GPU:** NVIDIA GeForce RTX 4070 (12 GB VRAM)
<!-- - **GPU (Integrated):** AMD Radeon (Integrated)
- **Memory:** 62 GB RAM
- **Storage:**
  - `/` : 280 GB SSD (ext4)
  - `/home` : 763 GB SSD (ext4)
- **Display:**
  - 1920×1080 @ 240 Hz (External)
  - 1920×1080 @ 60 Hz (External) -->

### Software
- **OS:** Ubuntu 22.04.5 LTS (Jammy Jellyfish) x86_64
- **Kernel:** Linux 6.8.0-87-generic
- **Desktop Environment:** GNOME 42.9 (X11, Mutter)
- **Shell:** bash 5.1.16
- **Python:** Python 3.10.x (via `venv`)

### CUDA / GPU Stack
- **NVIDIA Driver:** 570.195.03
- **CUDA Toolkit:** 12.8
- **NVCC:** V12.8.61
- **GPU Compute Mode:** Default
- **Persistence Mode:** Off

### Notes
- All experiments, simulations, and benchmarks reported in this repository were executed on the above system.
- CUDA-enabled training and inference were validated on the nvidia RTX 4070.
- Results may vary slightly across different hardware, driver versions, or operating systems.
