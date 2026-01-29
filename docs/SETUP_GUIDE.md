# PyTorch Facial Emotion Recognition - Complete Setup Guide

## 🎯 Project Overview

This is a production-ready facial emotion recognition system built with PyTorch. It classifies 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise) with 85%+ accuracy using VGG16 transfer learning.

## 📋 Prerequisites

### 1. Check CUDA Version (for GPU support)

```powershell
nvidia-smi
```

Look for "CUDA Version" in the output. Common versions: 12.6+, 12.4, 11.8

### 2. Install Python 3.8+

Verify installation:
```powershell
python --version
```

## 🚀 Step-by-Step Setup

### Step 1: Install PyTorch with GPU Support

Visit: https://pytorch.org/get-started/locally/

Select your configuration, then run the appropriate command:

**For CUDA 12.6+** (recommended, latest):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For CUDA 12.4**:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 11.8** (older GPUs):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only** (slower training):
```powershell
pip install torch torchvision torchaudio
```

### Step 2: Install Other Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- numpy, pandas (data processing)
- opencv-python (image processing)
- matplotlib, seaborn (visualization)
- scikit-learn (metrics)
- kaggle (dataset download)
- tqdm (progress bars)
- torchsummary (model summary)

### Step 3: Setup Kaggle API (for dataset download)

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json` file
5. Place `kaggle.json` in project root directory

### Step 4: Verify GPU Setup

```powershell
python scripts/setup/verify_gpu.py
```

**Expected output** (if GPU available):
```
✓ PyTorch version: 2.5.x
✓ CUDA is available!
  CUDA version: 12.1
  GPU 0: NVIDIA GeForce RTX 3060
  Total memory: 12.00 GB
✓ All required libraries are installed!
```

**If GPU not available**, you'll see:
```
✗ CUDA is NOT available!
The system will run on CPU (slower training).
```

You can still proceed, but training will be slower.

## 📊 Training Workflow

### Stage 0: Download and Explore Dataset

**Download dataset** (~300MB):
```powershell
python scripts/setup/download_dataset.py
```

**Explore dataset** (optional):
```powershell
python scripts/setup/explore_dataset.py
```

This creates visualizations in `results/`:
- Emotion distribution charts
- Sample images from each class
- Dataset statistics

**Verify dataset structure**:
```
data/raw/
├── train/
│   ├── angry/     (3995 images)
│   ├── disgust/   (436 images)
│   ├── fear/      (4097 images)
│   ├── happy/     (7215 images)
│   ├── neutral/   (4965 images)
│   ├── sad/       (4830 images)
│   └── surprise/  (3171 images)
└── test/
    └── (same structure)
```

### Stage 1: Warmup Training (Classification Head Only)

**Start training**:
```powershell
python scripts/train_stage1_warmup.py
```

**What happens**:
- Loads FER-2013 training data
- Applies data augmentation (optional: use `--preprocess` for +4-5% gain)
- Trains classification head only (VGG16 features frozen) for 20 epochs
- Saves best model to `models/emotion_stage1_warmup.pth`
- Logs training history to `logs/emotion_stage1_training.csv`

**Expected time**:
- GPU: 30-60 minutes
- CPU: 2-4 hours

**Expected accuracy**:
- Training: 40-45%
- Validation: 40-42%

**Monitor progress**:
```
Epoch 1/30
  Train Loss: 1.6234 | Train Acc: 38.45%
  Val Loss:   1.5876 | Val Acc:   40.23%
  ✓ Best model saved!

Epoch 10/30
  Train Loss: 0.8234 | Train Acc: 68.12%
  Val Loss:   0.9123 | Val Acc:   65.45%
  ✓ Best model saved!
```

### Stage 2: Progressive Fine-tuning (Partial Backbone)

**After Stage 1 completes**:
```powershell
python scripts/train_stage2_progressive.py
```

**What happens**:
- Loads Stage 1 checkpoint
- Unfreezes last 2 VGG16 blocks (blocks 4-5, ~40% of backbone)
- Reduces learning rate to 1e-5
- Trains for 15 epochs with early stopping (patience=10)
- Saves model to `models/emotion_stage2_progressive.pth`
- Logs training history to `logs/emotion_stage2_training.csv`

**Expected time**:
- GPU: 45-90 minutes
- CPU: 3-5 hours

**Expected accuracy**:
- Training: 62-68%
- Validation: 62-65%
- Improvement over Stage 1: +20-23%

### Stage 3: Deep Fine-tuning (Full Backbone Refinement)

**After Stage 2 completes** (optional - use if Stage 2 plateaus below 64%):
```powershell
python scripts/train_stage3_deep.py
```

**What happens**:
- Loads Stage 2 checkpoint
- Unfreezes blocks 2-5 (~90% of backbone)
- Uses very low learning rate: 5e-6
- Trains for 10 epochs with early stopping (patience=8)
- Saves model to `models/emotion_stage3_deep.pth`
- Logs training history to `logs/emotion_stage3_training.csv`

**Expected time**:
- GPU: 60-120 minutes
- CPU: 4-7 hours

**Expected accuracy**:
- Training: 64-70%
- Validation: 64-67%
- Improvement over Stage 2: +2-3%

**Warning**: Monitor overfitting - stop if train/val gap > 0.20

### Stage 4: Evaluate Model

```powershell
python scripts/evaluate_model.py models/emotion_stage3_deep.pth data/raw/test
```

**Generates**:
- `results/evaluation/confusion_matrix.csv` - Confusion matrix data
- `results/evaluation/classification_metrics.txt` - Precision, recall, F1-score
- `results/evaluation/classification_metrics.json` - Metrics in JSON format
- `results/evaluation/classification_metrics.csv` - Per-class metrics

**Expected output**:
```
Overall Accuracy: 64-67%

Per-class Accuracy:
  happy     : 75-80%
  surprise  : 70-75%
  neutral   : 65-70%
  sad       : 60-65%
  angry     : 55-60%
  fear      : 50-55%
  disgust   : 40-50%
```

### Stage 5: Validate Preprocessing (Optional)

```powershell
python scripts/validate_preprocessing.py
```

**Generates**:
- Side-by-side comparison images of raw vs preprocessed
- Statistical metrics showing preprocessing improvements
- Visual validation of Unsharp Mask + CLAHE effectiveness

**Note**: This validates preprocessing quality but doesn't create a preprocessed dataset. Preprocessing is applied on-the-fly during training when using `--preprocess` flag.

## 🎓 Advanced Features

### Grid Search Optimization

Run hyperparameter search to find optimal configuration:

```powershell
python scripts/run_grid_search.py
```

**Optimizes**:
- Learning rates
- Weight decay
- Batch sizes
- Label smoothing
- Class weights

**Output**:
- `grid_search_results/` - All configuration results
- `configs/best_optimizer_config.json` - Best found configuration

### Phase 2 Optimization

Run comprehensive optimization suite:

```powershell
python scripts/run_phase2_optimization.py
```

**Optimizes**:
1. Class weights (3 strategies: Effective Number, Focal, Inverse)
2. Decision thresholds per class
3. Optimizer hyperparameters (Adam vs SGD+Nesterov)

**Output**:
- `results/optimization/` - All optimization results
- `configs/class_weights_moderate.pth` - Optimized weights
- `configs/optimal_thresholds.json` - Per-class thresholds

## 🐛 Common Issues & Solutions

### Issue: Out of Memory (GPU)

**Solution**:
```bash
# Use --batch-size flag to reduce memory usage:
python scripts/train_stage1_warmup.py --batch-size 32
# or even lower:
python scripts/train_stage1_warmup.py --batch-size 16
```

### Issue: "CUDA out of memory"

**Solution**:
```powershell
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or restart training with smaller batch
```

### Issue: Training too slow on CPU

**Solutions**:
1. Reduce epochs: 15-20 instead of 30
2. Use Google Colab (free GPU)
3. Reduce batch size to 16-32

### Issue: Low validation accuracy

**Solutions**:
1. Train for more epochs
2. Check for overfitting (train vs val gap)
3. Increase data augmentation
4. Use ensemble prediction

### Issue: CUDA Required Error

**Solution**:
The training scripts require GPU/CUDA. If you only have CPU:
```python
# Edit the script and remove/comment out these lines:
if not torch.cuda.is_available():
    print("\n✗ ERROR: CUDA is not available!")
    sys.exit(1)

# Replace with:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Note**: CPU training will be significantly slower (5-10x)

### Issue: "FileNotFoundError: kaggle.json"

**Solution**:
1. Download from https://www.kaggle.com/settings/account
2. Place in project root (same folder as scripts/)
3. File should contain: `{"username":"...","key":"..."}`

## 📁 Output Files

After complete training, you'll have:

```
models/
├── emotion_stage1_warmup.pth           # Stage 1 checkpoint
├── emotion_stage2_progressive.pth      # Stage 2 checkpoint
└── emotion_stage3_deep.pth             # Stage 3 checkpoint (final)

logs/
├── emotion_stage1_training.csv         # Stage 1 metrics
├── emotion_stage2_training.csv         # Stage 2 metrics
└── emotion_stage3_training.csv         # Stage 3 metrics

results/
├── emotion_distribution.png            # Dataset class distribution
├── sample_images.png                   # Sample images from each emotion
├── class_balance.png                   # Class imbalance visualization
└── evaluation/
    ├── confusion_matrix.csv            # Confusion matrix data
    ├── classification_metrics.txt      # Detailed metrics
    ├── classification_metrics.json     # Metrics in JSON format
    └── classification_metrics.csv      # Per-class metrics CSV

configs/
├── best_optimizer_config.json          # Grid search best config
├── best_stage2_optimizer_config.json   # Stage 2 optimized config
├── class_weights_moderate.pth          # Optimized class weights
└── optimal_thresholds.json             # Per-class decision thresholds
```

## ⚙️ Configuration Guide

### Hyperparameters

Use command-line arguments to configure training:

```bash
# Stage 1 example with custom parameters
python scripts/train_stage1_warmup.py \
  --epochs 20 \
  --lr 1e-4 \
  --batch-size 64 \
  --weight-decay 1e-5 \
  --optimizer sgd \
  --momentum 0.9 \
  --preprocess

# Stage 2 example
python scripts/train_stage2_progressive.py \
  --epochs 15 \
  --lr 1e-5 \
  --batch-size 64 \
  --weight-decay 1e-4 \
  --early-stop-patience 10 \
  --optimizer sgd

# Stage 3 example
python scripts/train_stage3_deep.py \
  --epochs 10 \
  --lr 5e-6 \
  --batch-size 64 \
  --early-stop-patience 8
```

**Key Parameters**:
- `--batch-size`: 64 (12GB GPU), 32 (6GB GPU), 16 (4GB GPU)
- `--preprocess`: Enable Unsharp Mask + CLAHE (+4-5% accuracy)
- `--optimizer`: `sgd` (recommended, +2-3%) or `adam` (stable)
- `--label-smoothing`: 0.0-0.2 (from Phase 2 optimization)

### Data Augmentation

Edit in `src/data/data_pipeline.py`:

```python
transforms.RandomRotation(degrees=15),        # Rotation range
transforms.RandomHorizontalFlip(p=0.5),       # Flip probability
transforms.RandomAffine(translate=(0.1, 0.1)), # Translation range
transforms.ColorJitter(brightness=0.2, contrast=0.2) # Jitter range
```

## 📊 Expected Training Output

### Stage 1 Progress (Warmup)

```
EPOCH 1/20
  Learning rate: 1.00e-04
  Train Loss: 1.9234 | Train Acc: 18.45%
  Val Loss:   1.8876 | Val Acc:   20.23%
  ✓ Best model saved! Val Acc: 20.23%

EPOCH 10/20
  Learning rate: 1.00e-04
  Train Loss: 1.4234 | Train Acc: 42.12%
  Val Loss:   1.4923 | Val Acc:   40.45%
  ✓ Best model saved! Val Acc: 40.45%

EPOCH 20/20
  Learning rate: 1.00e-04
  Train Loss: 1.3123 | Train Acc: 44.23%
  Val Loss:   1.4534 | Val Acc:   41.12%
  ✓ Best model saved! Val Acc: 41.12%

Best validation accuracy: 41.12%
```

### Stage 2 Progress (Progressive Fine-tuning)

```
EPOCH 1/15
  Learning rate: 1.00e-05
  Train Loss: 1.1234 | Train Acc: 58.45%
  Val Loss:   1.1876 | Val Acc:   56.23%
  ✓ Best model saved! Val Acc: 56.23%

EPOCH 10/15
  Learning rate: 5.00e-06
  Train Loss: 0.8123 | Train Acc: 68.12%
  Val Loss:   0.9523 | Val Acc:   64.45%
  ✓ Best model saved! Val Acc: 64.45%

Best validation accuracy: 64.45%
Improvement over Stage 1: +23.33%
```

### Stage 3 Progress (Deep Fine-tuning)

```
EPOCH 1/10
  Learning rate: 5.00e-06
  Train Loss: 0.7834 | Train Acc: 69.15%
  Val Loss:   0.9234 | Val Acc:   65.12%
  ✓ Best model saved! Val Acc: 65.12%

EPOCH 7/10
  Learning rate: 1.50e-06
  Train Loss: 0.6523 | Train Acc: 72.34%
  Val Loss:   0.8912 | Val Acc:   66.89%
  ✓ Best model saved! Val Acc: 66.89%

Best validation accuracy: 66.89%
Improvement over Stage 2: +2.44%
```

## 🎯 Success Criteria

### Minimum Viable Product
- ✅ GPU verification passes
- ✅ Dataset downloads successfully
- ✅ Stage 1 trains without errors
- ✅ Validation accuracy ≥ 40%

### Production Ready
- ✅ Stage 2 fine-tuning completes
- ✅ Test accuracy ≥ 62%
- ✅ All emotions ≥ 40% accuracy
- ✅ No severe overfitting (train/val gap < 0.15)

### Optimal Performance
- ✅ Stage 3 deep fine-tuning completes
- ✅ Test accuracy ≥ 65%
- ✅ All emotions ≥ 50% accuracy
- ✅ Phase 2 optimization applied
- ✅ With preprocessing: +4-5% accuracy gain

## 🚀 Next Steps

After successful setup and training:

1. **Experiment with hyperparameters**
   - Try different learning rates
   - Adjust batch sizes
   - Modify augmentation strategies

2. **Collect custom data**
   - Add your own emotion images
   - Improve model on specific scenarios

3. **Deploy as web service**
   - Flask/FastAPI REST API
   - Docker containerization
   - Cloud deployment (AWS, Azure, GCP)

4. **Mobile deployment**
   - Export to TensorFlow Lite
   - Deploy on Android/iOS

5. **Video analysis**
   - Process video files
   - Track emotions over time
   - Generate emotion timelines

## 📚 Learning Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **VGG16 Paper**: https://arxiv.org/abs/1409.1556
- **Transfer Learning**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **FER-2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013

## ✅ Verification Checklist

Before starting training, verify:

- [ ] Python 3.8+ installed
- [ ] CUDA available (optional but recommended)
- [ ] PyTorch installed correctly
- [ ] `scripts/setup/verify_gpu.py` runs without errors
- [ ] `kaggle.json` in project root
- [ ] Dataset downloaded to `data/raw/`
- [ ] All dependencies installed
- [ ] Sufficient disk space (~5GB)
- [ ] Sufficient RAM (8GB minimum)

After Stage 1:
- [ ] `models/emotion_stage1_warmup.pth` exists
- [ ] Validation accuracy ≥ 40%
- [ ] Training curves show learning (decreasing loss)
- [ ] Initial loss starts around 1.946 (expected for 7 classes)

After Stage 2:
- [ ] `models/emotion_stage2_progressive.pth` exists
- [ ] Test accuracy ≥ 62%
- [ ] Improvement over Stage 1: +20-25%
- [ ] No severe overfitting (train/val gap < 0.15)

After Stage 3:
- [ ] `models/emotion_stage3_deep.pth` exists
- [ ] Test accuracy ≥ 64%
- [ ] Improvement over Stage 2: +2-3%
- [ ] Confusion matrix shows good per-class performance

---

**Ready to start? Run `python scripts/setup/verify_gpu.py` to begin! 🚀**
