# Google Cloud Platform Setup for MOSAIC Training

This guide covers setting up GCP for training MOSAIC models, including running multiple experiments in parallel.

---

## Quick SSH

```bash
# SSH into a GCP VM (default user)
gcloud compute ssh mosaic-v100 --zone=us-central1-a

# SSH as a specific user
gcloud compute ssh andrewyang@mosaic-v100 --zone=us-central1-a
```

Then: `cd andrew/MOSAIC && conda activate mosaic`

---

## 1. Install Google Cloud CLI

### macOS / Linux

```bash
# Download and run Google's installer
curl https://sdk.cloud.google.com | bash

# Restart your shell to load the new PATH
exec -l $SHELL

# Initialize gcloud (will open browser for authentication)
gcloud init
```

### Verify Installation

```bash
gcloud --version
# Google Cloud SDK 4xx.x.x
# bq 2.x.x
# core 2024.xx.xx
# gcloud-crc32c 1.x.x
# gsutil 5.xx
```

---

## 2. Create a GCP Project

You need a project before you can use any GCP services.

### Step 1: Create Project via Web Console

1. Go to: https://console.cloud.google.com/projectcreate

2. Fill in the form:
   - **Project name**: `MOSAIC Training` (display name, can be anything)
   - **Organization**: Select your organization or "No organization"
   - **Project ID**: Will be auto-generated (e.g., `mosaic-485719`)

3. Click **"Create"** and wait ~30 seconds

### Step 2: Find Your Project ID

**Important**: GCP assigns a unique Project ID (like `mosaic-485719`). This is different from the display name.

1. Go to: https://console.cloud.google.com/home/dashboard
2. Look at the **"Project info"** card
3. Copy the **Project ID** (not Project Name)

### Step 3: Set Project in Terminal

```bash
# Set YOUR project ID (the one from Step 2)
gcloud config set project mosaic-485719

# Verify it's set correctly
gcloud config get-value project
```

### Step 4: Link Billing Account

You cannot create GPU VMs without billing linked. New accounts get **$300 free credits**.

1. Go to: https://console.cloud.google.com/billing/projects
2. Find your project in the list
3. Click **"Change billing"** or the three dots menu (⋮)
4. Select your billing account (create one if needed)

### Step 5: Enable APIs

**Via Web Console (Recommended):**

1. **Compute Engine**: https://console.cloud.google.com/apis/library/compute.googleapis.com
   - Select your project in the top dropdown
   - Click **"Enable"**

2. **Cloud Storage**: https://console.cloud.google.com/apis/library/storage.googleapis.com
   - Click **"Enable"**

**Via CLI:**

```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

Note: CLI may fail with university/organization accounts due to permissions. Use web console if you get errors.

### Step 6: Set Default Region/Zone

```bash
# us-central1 typically has good GPU availability
gcloud config set compute/zone us-central1-a
gcloud config set compute/region us-central1
```

### Step 7: Check GPU Quota

Before creating GPU VMs, you need **two types of quota**:

**1. Global GPU Quota (GPUS_ALL_REGIONS)**
```bash
gcloud compute project-info describe --format="value(quotas[name=GPUS_ALL_REGIONS])"
```

If this shows `0.0`, you cannot create any GPU VMs even if you have regional quota.

**2. Regional GPU Quota**
```bash
gcloud compute regions describe us-central1 --format="table(quotas.filter(metric:NVIDIA))"
```

This shows per-GPU-type limits (T4, A100, etc.) for a specific region.

**If either quota is 0**, request an increase:
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter by `GPUS_ALL_REGIONS` (global) or specific GPU type (regional)
3. Select and click **"Edit Quotas"**
4. Request limit of 1 or more

**Note:** School/organization accounts may need an admin to approve quota changes.

---

## 3. Create a GPU VM for Training

### Find Available Images

Image names change over time. List current PyTorch GPU images:

```bash
gcloud compute images list --project=deeplearning-platform-release --filter="name~pytorch" --format="table(name,family)"
```

### Option A: Single Command (T4 GPU - Cost Effective)

```bash
gcloud compute instances create mosaic-t4 \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

If the image is not found, run the list command above to find current image names.

### Option B: A100 GPU (Faster Training)

**Note:** A100 GPUs require the `a2-highgpu-*` machine family (not `n1-standard-*`). The `a2-highgpu-1g` provides 1 A100 (40GB), 12 vCPUs, and 85GB RAM.

A100 GPUs have limited availability. Check which zones have A100:

```bash
gcloud compute accelerator-types list --filter="name=nvidia-tesla-a100"
```

Common zones with A100:

```
nvidia-tesla-a100  us-central1-a      NVIDIA A100 40GB
nvidia-tesla-a100  us-central1-b      NVIDIA A100 40GB
nvidia-tesla-a100  us-central1-c      NVIDIA A100 40GB
nvidia-tesla-a100  us-central1-f      NVIDIA A100 40GB
nvidia-tesla-a100  us-west1-b         NVIDIA A100 40GB
nvidia-tesla-a100  us-east1-b         NVIDIA A100 40GB
nvidia-tesla-a100  asia-northeast1-a  NVIDIA A100 40GB
nvidia-tesla-a100  asia-northeast1-c  NVIDIA A100 40GB
nvidia-tesla-a100  asia-southeast1-b  NVIDIA A100 40GB
nvidia-tesla-a100  asia-southeast1-c  NVIDIA A100 40GB
nvidia-tesla-a100  europe-west4-b     NVIDIA A100 40GB
nvidia-tesla-a100  europe-west4-a     NVIDIA A100 40GB
nvidia-tesla-a100  asia-northeast3-a  NVIDIA A100 40GB
nvidia-tesla-a100  asia-northeast3-b  NVIDIA A100 40GB
nvidia-tesla-a100  us-west3-b         NVIDIA A100 40GB
nvidia-tesla-a100  us-west4-b         NVIDIA A100 40GB
nvidia-tesla-a100  me-west1-a         NVIDIA A100 40GB
nvidia-tesla-a100  me-west1-c         NVIDIA A100 40GB
```

```bash
gcloud compute instances create mosaic-a100 \
  --zone=europe-west4-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --metadata="install-nvidia-driver=True"
```

If you get "zone does not have enough resources", try a different zone from the list above.

### Option C: Preemptible/Spot VM (60-70% Cheaper)

```bash
gcloud compute instances create mosaic-spot \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --metadata="install-nvidia-driver=True"
```

**Warning**: Spot VMs can be preempted (terminated) at any time. Use checkpointing!

### GPU Cost Reference

| GPU | $/hour (Standard) | $/hour (Spot) | Use Case |
|-----|-------------------|---------------|----------|
| T4 | ~$0.35 | ~$0.11 | Development, small experiments |
| V100 | ~$2.48 | ~$0.74 | Serious training |
| A100 | ~$4.00 | ~$1.20 | Large models, fast training |

---

## 4. Connect to Your VM

### List Your VMs

```bash
gcloud compute instances list
```

This shows all your VMs with name, zone, machine type, IPs, and status (RUNNING/STOPPED).

### SSH into the VM

```bash
gcloud compute ssh mosaic-a100 --zone=europe-west4-a
```

If you can't ssh into it, it's probably an meta-info loss issue, just reset:

```bash
gcloud compute instances reset mosaic-a100 --zone=europe-west4-a
```

### Verify GPU is Available

```bash
nvidia-smi
# Should show your GPU (T4, A100, etc.)
```

---

## 5. Setup MOSAIC Environment on VM

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/KevinBian107/MOSAIC.git
cd MOSAIC

# Check if conda is available
conda --version

# If not installed, install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment from server config (optimized for CUDA 12.x)
conda env create -f environment_server.yaml
conda activate mosaic

# Verify PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 6. Run Training

### Single Training Run

```bash
# Inside tmux session
cd ~/MOSAIC
conda activate mosaic

python scripts/train.py \
  tokenizer=hdt \
  data.dataset_name=moses \
  model.model_name=gpt2-s \
  training.max_steps=100000
```

### Auto-Stop VM After Training (Save Money!)

To automatically shut down the VM when training completes (so you don't keep paying):

```bash
# Run training, then auto-shutdown when done
python scripts/train.py \
  tokenizer=hdt \
  data.dataset_name=moses \
  data.num_train=50000 \
  && sudo shutdown -h now
```

The `&& sudo shutdown -h now` part only runs if training completes successfully. The VM will stop and you won't be billed for compute (only minimal disk storage ~$10/month).

**Important:** Make sure to sync your results before shutdown, or save checkpoints to Cloud Storage:

```bash
# Save results to Cloud Storage before shutdown
python scripts/train.py ... \
  && gsutil -m cp -r ~/MOSAIC/outputs gs://YOUR_BUCKET/results/ \
  && sudo shutdown -h now
```

### Monitor Training

```bash
# In another tmux window (Ctrl+B, then C for new window)
cd ~/MOSAIC
tail -f outputs/train/*/train.log

# Or use tensorboard
tensorboard --logdir outputs/train --port 6006
```

To access TensorBoard from your local machine:
```bash
# On your local machine, create SSH tunnel
gcloud compute ssh mosaic-t4 --zone=us-central1-a -- -L 6006:localhost:6006

# Then open http://localhost:6006 in your browser
```

---

## 7. Running Multiple Training Jobs in Parallel

### Method 1: Multiple tmux Windows (Same VM)

If your VM has enough GPU memory for multiple small jobs.

**tmux Quick Reference (Mac: use Control key, not Command):**

| Action | Shortcut |
|--------|----------|
| New window | `Control+B`, release, then `C` |
| Previous window | `Control+B`, release, then `P` |
| Next window | `Control+B`, release, then `N` |
| Go to window 0/1/2 | `Control+B`, release, then `0`/`1`/`2` |
| List all windows | `Control+B`, release, then `W` |
| Detach session | `Control+B`, release, then `D` |

```bash
# Create named tmux session
tmux new -s experiments

# Download MOSES data (one-time setup)
conda activate mosaic
mkdir -p ~/MOSAIC/data/moses
cd ~/MOSAIC/data/moses
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv
cd ~/MOSAIC

# ===== Window 0: SENT tokenizer (flat, no hierarchy) =====
python scripts/train.py tokenizer=sent data.num_train=500000 resume=False

# ===== Window 1: HDT with motif_community coarsening =====
# Create new window: Control+B, then C
# Activate env in new window:
conda activate mosaic && cd ~/MOSAIC
python scripts/train.py tokenizer=hdt tokenizer.coarsening_strategy=motif_community data.num_train=500000 resume=False

# ===== Window 2: HDT with spectral coarsening =====
# Create new window: Control+B, then C
conda activate mosaic && cd ~/MOSAIC
python scripts/train.py tokenizer=hdt tokenizer.coarsening_strategy=spectral data.num_train=500000 resume=False

# ===== Window 3: HSENT with motif_community coarsening =====
# Create new window: Control+B, then C
conda activate mosaic && cd ~/MOSAIC
python scripts/train.py tokenizer=hsent tokenizer.coarsening_strategy=motif_community data.num_train=500000 resume=False

# ===== Window 4: HSENT with spectral coarsening =====
# Create new window: Control+B, then C
conda activate mosaic && cd ~/MOSAIC
python scripts/train.py tokenizer=hsent tokenizer.coarsening_strategy=spectral data.num_train=500000 resume=False

# ===== Window 5: HDTC tokenizer (compositional, uses functional groups) =====
# Create new window: Control+B, then C
conda activate mosaic && cd ~/MOSAIC
python scripts/train.py tokenizer=hdtc data.num_train=500000 resume=False

# To check on experiments: Control+B, then W to list all windows
# To detach and leave running: Control+B, then D
# To reattach later: tmux attach -t experiments
```

**Full experiment matrix:**

| Tokenizer | Coarsening Strategy | Command Override |
|-----------|---------------------|------------------|
| SENT | N/A (flat) | `tokenizer=sent` |
| HDT | motif_community | `tokenizer=hdt tokenizer.coarsening_strategy=motif_community` |
| HDT | spectral | `tokenizer=hdt tokenizer.coarsening_strategy=spectral` |
| HSENT | motif_community | `tokenizer=hsent tokenizer.coarsening_strategy=motif_community` |
| HSENT | spectral | `tokenizer=hsent tokenizer.coarsening_strategy=spectral` |
| HDTC | N/A (functional groups) | `tokenizer=hdtc` |

### Method 2: Multiple VMs (Recommended for Full Training)

Create multiple VMs and run one experiment per VM:

```bash
# Create 6 VMs for parallel experiments (one per tokenizer config)
for i in 1 2 3 4 5 6; do
  gcloud compute instances create mosaic-exp-$i \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --metadata="install-nvidia-driver=True"
done
```

---

## 8. Syncing Results Back to Local

### Copy Files from VM

```bash
# Copy single file
gcloud compute scp mosaic-t4:~/MOSAIC/outputs/train/*/best.ckpt ./results/ --zone=us-central1-a

# Copy entire output directory
gcloud compute scp --recurse mosaic-t4:~/MOSAIC/outputs ./local_outputs --zone=us-central1-a
```

### Using Cloud Storage (Recommended for Large Files)

```bash
# On VM: Upload results to Cloud Storage
gsutil -m cp -r ~/MOSAIC/outputs gs://YOUR_BUCKET/mosaic-results/

# On local: Download from Cloud Storage
gsutil -m cp -r gs://YOUR_BUCKET/mosaic-results/ ./results/
```

Setup Cloud Storage bucket:
```bash
# Create bucket (one-time)
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME/
```

---

## 9. VM Management

### Stop VM (Pause Billing)

```bash
gcloud compute instances stop mosaic-t4 --zone=us-central1-a
```

### Start VM Again

```bash
gcloud compute instances start mosaic-t4 --zone=us-central1-a
```

### Delete VM (Remove Completely)

```bash
gcloud compute instances delete mosaic-t4 --zone=us-central1-a
```

### List All VMs

```bash
gcloud compute instances list
```

### Stop All MOSAIC VMs

```bash
gcloud compute instances list --filter="name~mosaic" --format="value(name,zone)" | \
  while read name zone; do
    gcloud compute instances stop $name --zone=$zone
  done
```

---

## 10. Cost Management Tips

### 1. Always Stop VMs When Not Training

```bash
# Check running instances
gcloud compute instances list --filter="status=RUNNING"

# Stop all running MOSAIC VMs
gcloud compute instances stop $(gcloud compute instances list --filter="name~mosaic AND status=RUNNING" --format="value(name)") --zone=europe-west4-a
```

### 2. Use Spot/Preemptible VMs for Long Training

60-70% cost savings, but implement checkpointing:

```python
# In your training config, save checkpoints frequently
training:
  checkpoint_every_n_steps: 1000
  save_top_k: 3
```

### 3. Set Budget Alerts

```bash
# Create budget alert (via console is easier)
# https://console.cloud.google.com/billing/budgets
```

### 4. Use Smaller VMs for Development

```bash
# Cheap development VM (no GPU)
gcloud compute instances create mosaic-dev \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB
```

---

## Troubleshooting

### "PERMISSION_DENIED" Errors from CLI

```
ERROR: (gcloud.services.enable) PERMISSION_DENIED: Permission denied
```

**Most common cause:** Wrong project ID set. GCP assigns a unique Project ID (like `mosaic-485719`) that's different from the display name.

**Fix:**
1. Find your Project ID: https://console.cloud.google.com/home/dashboard (look at "Project info" card)
2. Set the correct project:
   ```bash
   gcloud config set project YOUR-ACTUAL-PROJECT-ID
   ```

**Other causes:**
- Billing not linked: https://console.cloud.google.com/billing/projects
- University/org account restrictions: Try using a personal Gmail account instead

**Workaround:** Enable APIs via web console instead of CLI:
- https://console.cloud.google.com/apis/library/compute.googleapis.com

### "Image not found" Error

```
ERROR: The resource 'projects/deeplearning-platform-release/global/images/family/...' was not found
```

Image names change over time. List current available images:

```bash
gcloud compute images list --project=deeplearning-platform-release --filter="name~pytorch" --format="table(name,family)"
```

Use a family name from the output, e.g.:
- `--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570` (Ubuntu 22.04)
- `--image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570` (Ubuntu 24.04)

### "Quota exceeded" or "GPUS_ALL_REGIONS" Error

```
ERROR: Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```

You need **both** global and regional GPU quota. Check both:

```bash
# Check global quota (required!)
gcloud compute project-info describe --format="value(quotas[name=GPUS_ALL_REGIONS])"

# Check regional quota
gcloud compute regions describe us-central1 --format="table(quotas.filter(metric:NVIDIA))"
```

If global quota is 0, request increase at: https://console.cloud.google.com/iam-admin/quotas
- Filter by `GPUS_ALL_REGIONS`
- Request limit of 1

For school/organization accounts, contact your GCP admin to increase the quota.

### "Zone does not have enough resources" Error

```
ERROR: The zone 'projects/.../zones/us-central1-a' does not have enough resources available
```

The requested GPU is not available in that zone. **Solution:**

1. Check which zones have your GPU type:
   ```bash
   # For A100
   gcloud compute accelerator-types list --filter="name=nvidia-tesla-a100"

   # For T4
   gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"
   ```

2. Try a different zone from the list:
   - A100 common zones: `us-west1-a`, `us-west1-b`, `us-east1-c`, `europe-west4-a`
   - T4 is more widely available

3. Or switch to a different GPU type (T4 is more available than A100)

### VM Won't Start (Spot Instance)

Spot instances may not be available. Try:
1. Different zone: `--zone=us-central1-b`
2. Different GPU: `nvidia-tesla-t4` instead of `a100`
3. Standard VM (more expensive): Remove `--provisioning-model=SPOT`

### SSH Connection Timeout

```bash
# Use IAP tunnel instead
gcloud compute ssh VM_NAME --zone=us-central1-a --tunnel-through-iap
```

### NVIDIA Driver Not Found / nvidia-smi Fails

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

**Step 1: Verify the GPU is attached**

```bash
lspci | grep -i nvidia
# Should show something like:
# 00:04.0 3D controller: NVIDIA Corporation GV100GL [Tesla A100 SXM2 16GB]
```

If no output, the GPU is not attached to the VM. Recreate the VM with GPU attached.

**Step 2: Check what NVIDIA packages are installed**

```bash
dpkg -l | grep -i nvidia
```

**Step 3a: If no drivers installed, install them**

```bash
sudo apt update
sudo apt install -y nvidia-driver-535-server
sudo reboot
```

**Step 3b: If partial 570-server packages exist (common with Deep Learning VMs)**

If you see packages like `nvidia-kernel-common-570-server`, `libnvidia-compute-570-server`, etc. but `nvidia-smi` still fails, complete the installation:

```bash
# Install the matching driver version
sudo apt install -y nvidia-driver-570-server

# Or for the open kernel module variant
sudo apt install -y nvidia-driver-570-server-open

# Reboot to load the driver
sudo reboot
```

**Step 4: If installation fails with package conflicts**

```bash
# Fix broken packages first
sudo apt --fix-broken install

# Remove conflicting packages if needed
sudo apt remove --purge nvidia-kernel-common* nvidia-dkms* -y
sudo apt autoremove -y

# Then retry installation
sudo apt install -y nvidia-driver-570-server
```

**Step 5: Try the Deep Learning VM driver installer (if available)**

```bash
sudo /opt/deeplearning/install-driver.sh
```

**Step 6: Verify after reboot**

```bash
nvidia-smi
# Should show your GPU with driver version
```

### PyTorch Can't See GPU (nvidia-smi works but torch.cuda.is_available() is False)

```
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment
```

This happens when `nvidia-smi` works but PyTorch cannot initialize CUDA. Usually caused by the driver kernel module not being fully loaded.

**Solution: Reboot the VM**

```bash
sudo reboot
```

After reboot, SSH back in and verify:

```bash
gcloud compute ssh mosaic-a100 --zone=europe-west4-a
conda activate mosaic
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True and Tesla A100-SXM2-16GB
```