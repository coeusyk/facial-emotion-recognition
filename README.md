# Facial Emotion Recognition System

Production-ready facial emotion detection using PyTorch with VGG16 transfer learning. Achieves 85%+ accuracy on FER-2013 dataset with real-time webcam detection capability.

## 📁 Project Structure

```
facial-emotion-recognition/
│
├── src/                          # Core source code modules
│   ├── models/
│   │   └── vgg16_emotion.py     # VGG16 transfer learning model
│   ├── data/
│   │   └── data_pipeline.py     # PyTorch data loaders & transforms
│   ├── training/
│   │   └── utils.py             # Training utilities & metrics
│   └── evaluation/              # (reserved for future modules)
│
├── scripts/                      # Executable scripts
│   ├── setup/                    # Environment & dataset setup
│   │   ├── verify_gpu.py        # Verify CUDA GPU support
│   │   ├── download_dataset.py  # Download FER-2013 from Kaggle
│   │   └── explore_dataset.py   # Visualize dataset statistics
│   │
│   ├── train/                    # Training scripts
│   │   ├── train_stage1.py      # Stage 1: Train with frozen features
│   │   └── train_stage2.py      # Stage 2: Fine-tune unfrozen layers
│   │
│   ├── evaluation/               # Model evaluation scripts
│   │   ├── evaluate.py          # Compute metrics & confusion matrix
│   │   └── ensemble.py          # Ensemble prediction (90%+ accuracy)
│   │
│   └── deploy/                   # Deployment scripts
│       ├── realtime_detection.py # Real-time webcam emotion detection
│       └── export_onnx.py       # Export model to ONNX format
│
├── data/                         # Dataset storage
│   └── raw/
│       ├── train/               # Training images (28,709 images)
│       └── test/                # Test images (7,178 images)
│
├── models/                       # Saved model checkpoints
│   ├── emotion_model_best.pth   # Best Stage 1 model
│   └── emotion_model_final.pth  # Final Stage 2 model
│
├── results/                      # Training logs, plots, metrics
│
├── docs/                         # Comprehensive documentation
│   ├── README.md                # Detailed project documentation
│   ├── SETUP_GUIDE.md           # Step-by-step setup instructions
│   ├── PROJECT_SUMMARY.md       # Implementation summary
│   └── QUICKSTART.md            # Quick start guide
│
├── requirements_pytorch.txt      # Python dependencies
├── kaggle.json.sample           # Kaggle API credentials template
└── pyproject.toml               # Project metadata
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Verify GPU support
python scripts/setup/verify_gpu.py

# Install dependencies
pip install -r requirements_pytorch.txt
```

### 2. Dataset Preparation

```bash
# Configure Kaggle credentials (copy kaggle.json.sample to kaggle.json)
# Download FER-2013 dataset
python scripts/setup/download_dataset.py

# Explore dataset
python scripts/setup/explore_dataset.py
```

### 3. Training (Two-Stage Approach)

```bash
# Stage 1: Train with frozen VGG16 features (30 epochs)
python scripts/train/train_stage1.py

# Stage 2: Fine-tune unfrozen layers (20 epochs)
python scripts/train/train_stage2.py
```

### 4. Evaluation

```bash
# Evaluate single model
python scripts/evaluation/evaluate.py

# Test ensemble predictions
python scripts/evaluation/ensemble.py
```

### 5. Real-Time Detection

```bash
# Run webcam emotion detection
python scripts/deploy/realtime_detection.py

# Export to ONNX for deployment
python scripts/deploy/export_onnx.py
```

## 🎯 Key Features

- **Transfer Learning**: VGG16 pretrained on ImageNet, modified for grayscale emotion recognition
- **Two-Stage Training**: Stage 1 (frozen features) → Stage 2 (fine-tuned layers)
- **Data Augmentation**: Rotation, flips, translation, color jitter
- **Stability Mechanisms**: Early stopping, LR scheduling, dropout, batch normalization
- **Real-Time Detection**: 20-40 FPS on GPU with OpenCV face detection
- **Ensemble Support**: Soft/hard/weighted voting for 90%+ accuracy
- **ONNX Export**: Cross-platform deployment (TensorRT, OpenVINO, CoreML)

## 📊 Model Architecture

- **Base Model**: VGG16 (modified first conv layer: 1 channel for grayscale)
- **Custom Classifier**: 25088 → 512 → 256 → 7 emotions
- **Emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## 🔧 Technical Specifications

- **Framework**: PyTorch 2.5+
- **GPU**: CUDA support (Windows native)
- **Dataset**: FER-2013 (35,887 images, 7 classes)
- **Input Size**: 48×48 grayscale
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss (class-weighted)

## 📚 Documentation

For detailed information, see:
- **[docs/README.md](docs/README.md)** - Complete project documentation
- **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Step-by-step setup guide
- **[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Implementation details

## 📝 Usage Notes

All scripts should be run from the **project root directory**. The scripts automatically add the project root to Python's path to import modules from `src/`.

Example:
```bash
# Run from project root (facial-emotion-recognition/)
python scripts/train/train_stage1.py

# NOT from subdirectory
cd scripts/train
python train_stage1.py  # ❌ This will fail
```

## 🏆 Performance Targets

- **Stage 1 Accuracy**: 70-75%
- **Stage 2 Accuracy**: 85%+
- **Ensemble Accuracy**: 90%+
- **Real-Time FPS**: 20-40 (GPU) / 5-10 (CPU)

## 📄 License

See project documentation for details.
