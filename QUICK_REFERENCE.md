# Quick Reference - PyTorch Facial Emotion Recognition

## 🚀 Essential Commands (In Order)

```powershell
# 1. Verify GPU and dependencies
python scripts/setup/verify_gpu.py

# 2. Download FER-2013 dataset (~300MB)
python scripts/setup/download_dataset.py

# 3. [NEW] Analyze and preprocess data
python scripts/data/preprocess_data.py

# 4. Train Stage 1 (1-2 hours GPU, 5-8 hours CPU)
python scripts/train/train_stage1.py

# 5. Train Stage 2 (30-60 min GPU, 2-4 hours CPU)
python scripts/train/train_stage2.py

# 6. Evaluate model performance
python scripts/evaluation/evaluate.py

# 7. Run real-time webcam detection
python scripts/deploy/realtime_detection.py
```

## 📊 Optional Commands

```powershell
# Explore dataset statistics
python scripts/setup/explore_dataset.py

# Verify data integrity
python scripts/setup/verify_data.py

# [DEPRECATED] Balance dataset - use weighted loss instead
# python scripts/data/balance_dataset.py

# Test ensemble prediction
python scripts/evaluation/ensemble.py

# Export to ONNX format
python scripts/deploy/export_onnx.py --benchmark

# Real-time detection with specific camera
python scripts/deploy/realtime_detection.py --camera 1

# Force CPU inference
python scripts/deploy/realtime_detection.py --cpu
```

## 🔧 Quick Fixes

### Out of Memory
```python
# Edit train_emotion_model.py or finetune_emotion_model.py
BATCH_SIZE = 32  # Instead of 64
NUM_WORKERS = 2  # Instead of 4
```

### Check GPU Status
```powershell
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### View Training Progress
```powershell
# Stage 1 logs
type logs\emotion_model_stage1_training.csv

# Stage 2 logs
type logs\emotion_model_stage2_training.csv
```

### Clear GPU Memory
```powershell
python -c "import torch; torch.cuda.empty_cache()"
```

## 📁 Important File Locations

```
Models:
  models/emotion_model_best.pth           # Stage 1 best
  models/emotion_model_final.pth          # Stage 2 final (USE THIS)
  
Logs:
  logs/emotion_model_stage1_training.csv  # Stage 1 metrics
  logs/emotion_model_stage2_training.csv  # Stage 2 metrics
  
Results:
  results/confusion_matrix.png            # Evaluation heatmap
  results/classification_report.txt       # Detailed metrics
  results/preprocessing_report.txt        # [NEW] Data preprocessing stats
  results/train_class_distribution.png    # [NEW] Class distribution
  results/preprocessing_comparison.png    # [NEW] Before/after preprocessing
```

## 🎯 Expected Accuracy

| Stage | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| Stage 1 | 70-75% | 68-72% | 67-71% |
| Stage 2 | 85-88% | 83-87% | 82-86% |
| Ensemble | - | - | 88-92% |

## ⏱️ Expected Training Time

| Hardware | Stage 1 | Stage 2 | Total |
|----------|---------|---------|-------|
| RTX 3060 | 1-2 hrs | 30-60 min | 1.5-3 hrs |
| GTX 1660 | 2-3 hrs | 1-1.5 hrs | 3-4.5 hrs |
| CPU (i7) | 5-8 hrs | 2-4 hrs | 7-12 hrs |

## 🎮 Webcam Controls

```
q - Quit
s - Save current frame
p - Pause/Resume
```

## 📋 Troubleshooting Quick Ref

| Error | Quick Fix |
|-------|-----------|
| CUDA out of memory | Reduce BATCH_SIZE to 32 or 16 |
| Webcam not opening | Try --camera 1 or --camera 2 |
| kaggle.json not found | Download from kaggle.com/settings/account |
| Low accuracy | Train longer or use ensemble |
| Slow training | Use GPU or reduce epochs |

## 🔗 Quick Links

- **PyTorch Install**: https://pytorch.org/get-started/locally/
- **Kaggle API Setup**: https://www.kaggle.com/settings/account
- **FER-2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **Preprocessing Guide**: See docs/PREPROCESSING_GUIDE.md
- **Full Documentation**: See README.md
- **Setup Guide**: See docs/SETUP_GUIDE.md

## 📞 Common Questions

**Q: Do I need a GPU?**
A: No, but recommended. CPU training is 3-5x slower.

**Q: How much disk space needed?**
A: ~2GB (300MB dataset + 500MB models + logs)

**Q: Can I stop training and resume?**
A: Yes, models are saved at each epoch. Load checkpoint to resume.

**Q: What if Stage 1 accuracy is low?**
A: Normal! Stage 2 improves by 10-15%. Target is 83-87% after Stage 2.

**Q: How to improve accuracy?**
A: 1) Train ensemble (90%+), 2) More epochs, 3) Better augmentation

**Q: Can I use custom images?**
A: Yes! Add to data/raw/train/ and data/raw/test/ folders.

---

**Need help? Check SETUP_GUIDE.md for detailed instructions!**
