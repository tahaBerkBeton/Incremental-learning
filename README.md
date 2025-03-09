# Incremental Learning with Mixture-of-Experts (MoE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB-green.svg)](https://benchmark.ini.rub.de/gtsrb_dataset.html)

## 📚 Overview

This repository implements an incremental learning system that combats catastrophic forgetting using a Mixture-of-Experts (MoE) approach with an improved LeNet-5 architecture. The system learns to classify traffic signs sequentially without forgetting previously learned classes, evaluated on the German Traffic Sign Recognition Benchmark (GTSRB).

<p align="center">
  <img src="architecture.png" alt="MoE Architecture" width="700"/>
  <br>
  <em>Architecture of the Mixture-of-Experts model with LeNet-5 feature extractor</em>
</p>

## 🔍 Problem Statement

**Catastrophic forgetting** is a fundamental challenge in machine learning where neural networks tend to forget previously learned information when trained on new data. Traditional approaches require retraining on all data, which becomes computationally expensive as datasets grow.

Our approach tackles this by:
- 🏗️ Employing a modular architecture that adds experts for new tasks
- 🔄 Preserving a frozen feature extractor to maintain learned representations
- 🧠 Using a memory buffer to retain examples from previous tasks
- 🧭 Implementing a router mechanism to direct inputs to the appropriate expert

## 🏛️ Model Architecture

### Feature Extractor
- Based on LeNet-5 with improvements (BatchNorm, Dropout)
- Processes RGB images (32×32 pixels)
- Extracts 400-dimensional feature vectors
- Frozen after initial training to preserve representations

```python
class LeNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # [B, 400]
```

### Expert Networks
- Task-specific networks responsible for their subset of classes
- Each expert includes:
  - Two fully-connected layers (400→120→84)
  - Dropout for regularization
  - Output layer sized according to the number of classes

### Router Network
- Linear layer that maps feature vectors to expert selection logits
- Selects the most appropriate expert for a given input
- Trained with cross-entropy loss to align with ground truth expert assignments

## 🔄 Incremental Learning Process

### Task Division
The GTSRB dataset is divided into 4 sequential tasks, with each introducing new traffic sign classes:
- Task 1: Classes 0-10
- Task 2: Classes 11-21
- Task 3: Classes 22-32
- Task 4: Classes 33-42

### Training Procedure
1. **First Task**: Train feature extractor and first expert (120 epochs)
2. **Subsequent Tasks**:
   - Freeze feature extractor and previous experts
   - Add new expert for current task classes
   - Train for 80 epochs with constant learning rate (0.001)
   - Incorporate memory buffer containing examples from previous tasks
   - Apply multi-component loss function:
     - Classification loss on new task data
     - Classification loss on memory buffer (weighted by 2.0)
     - Routing alignment loss (weighted by 0.1)
   - Save best model checkpoint based on validation accuracy

### Memory Buffer Strategy
- Maintains 1000 samples from previous tasks 
- Balanced allocation across past classes
- Randomly selected from test split to simulate real-world scenarios
- Replayed during training to mitigate forgetting

## 📊 Results

The incremental learning performance demonstrates the model's ability to learn new tasks while retaining knowledge from previous ones:

| Task | Classes | Accuracy |
|------|---------|----------|
| 1    | 0-10    | 83.26%   |
| 2    | 0-21    | 82.94%   |
| 3    | 0-32    | 75.74%   |
| 4    | 0-42    | 75.84%   |

While there is some performance degradation as new tasks are added, the model maintains a respectable accuracy across all classes, demonstrating effective mitigation of catastrophic forgetting.

## 🗂️ Directory Structure

```
incremental-learning/
├── MOE_Lenet.py          # Training script implementing incremental learning with MoE
├── inference_Lenet.py    # Inference script for evaluating the trained MoE model
├── architecture.png      # Visualization of the model architecture
└── checkpoints/          # Directory containing model checkpoints
    ├── moe_model_task1.pt
    ├── moe_model_task2.pt
    ├── moe_model_task3.pt
    └── moe_model_task4.pt
```

## 🛠️ Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.0
pillow>=8.2.0
tqdm>=4.60.0
```

## 🚀 How to Run

### Setup
```bash
# Clone repository
git clone https://github.com/username/incremental-learning-moe.git
cd incremental-learning-moe

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Run training script
python MOE_Lenet.py
```
Training creates checkpoints in the `checkpoints/` directory after completing each task.

### Inference
```bash
# Run inference (by default uses the task 4 model)
python inference_Lenet.py

# Specify a different checkpoint
python inference_Lenet.py --checkpoint ./checkpoints/moe_model_task3.pt
```

## 📝 Implementation Details

Key hyperparameters:
- Learning rate: 0.001 (constant)
- Batch size: 64
- Memory buffer size: 1000 samples
- Alignment strength: 0.1
- Buffer weight: 2.0

Loss function components:
```python
total_loss = (classification_loss_new +
              buffer_weight * classification_loss_buf +
              alignment_strength * (routing_loss_new + routing_loss_buf))
```

## 🔮 Future Work

- 🧪 Experiment with different router mechanisms that improve expert selection
- 🔍 Explore distillation methods to further reduce forgetting


## 📚 References

- German Traffic Sign Recognition Benchmark (GTSRB): https://benchmark.ini.rub.de/gtsrb_dataset.html
- "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al.
- "Mixture of Experts: A Literature Survey" by Masoudnia and Ebrahimpour

## 🙏 Acknowledgements

This project leverages several key resources:
- GTSRB Dataset for training and evaluation
- PyTorch framework for implementation


## 📄 License

To come...
