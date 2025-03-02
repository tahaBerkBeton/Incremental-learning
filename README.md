
---

# Incremental Learning with Mixture-of-Experts (MoE) using Improved LeNet-5

## Overview

This project implements an incremental learning system for traffic sign recognition using a Mixture-of-Experts (MoE) model based on an improved version of LeNet-5. The main goal is to demonstrate how to learn new classes sequentially without catastrophic forgetting, by progressively adding new experts for new tasks while retaining the knowledge from previous tasks. The system is evaluated on the German Traffic Sign Recognition Benchmark (GTSRB).

## Problem Statement

Incremental learning poses the challenge of updating a model with new information while retaining performance on previously learned tasks. In this context, the task is to classify traffic signs incrementally. Traditional neural networks tend to forget old tasks when retrained with new data—a phenomenon known as catastrophic forgetting. Our approach tackles this issue by:
- Dividing the learning process into multiple tasks.
- Adding dedicated experts for each new set of classes.
- Using a router to select the most relevant expert for a given input.
- Incorporating a memory buffer that maintains a balanced subset of data from past tasks.

## Strategy Adopted

### Mixture-of-Experts Architecture

The model architecture is divided into three main components:

1. **Feature Extractor**  
   An improved LeNet-5 based network processes input images (resized to 32×32 pixels) and extracts a 400-dimensional feature representation. This network is frozen after initial training to help preserve learned features.

2. **Expert Networks**  
   For each new incremental task, a new expert (a fully connected network) is added. Each expert is responsible for classifying a specific subset of traffic sign classes. By specializing, experts help maintain performance on previously learned tasks while accommodating new classes.

3. **Router Network**  
   A simple linear layer acts as the router. It takes the extracted features and outputs logits corresponding to each expert. The expert with the highest logit is selected to perform the classification, ensuring that the most appropriate expert handles each input.

### Incremental Learning Procedure

- **Task Division:**  
  The GTSRB dataset is split into 4 tasks, with each task containing a subset of traffic sign classes. As each new task arrives, the current set of classes is extended.

- **Training Process:**  
  - The first task is trained longer (120 epochs) compared to subsequent tasks (80 epochs) using a constant learning rate of 0.001.
  - A memory buffer is maintained that stores a fixed number of samples from previous tasks. This helps mitigate the forgetting of earlier tasks.
  - After training each task, the best model (as measured by evaluation accuracy) is checkpointed.

- **Evaluation:**  
  The model is evaluated on the full test dataset after each task. The best global performance achieved is 75.84% accuracy on Task 4.

## Results

The incremental learning performance across tasks is summarized below:

- **Task 1:** 83.26%
- **Task 2:** 82.94%
- **Task 3:** 75.74%
- **Task 4:** 75.84% *(Current best global performance)*

These results reflect the challenge of maintaining performance as new classes are added. While initial tasks show higher accuracy, the performance on later tasks typically decreases due to the increased difficulty of balancing new learning with the retention of previous knowledge.

## Directory Structure

The project directory is organized as follows:

```
incremental-learning/
├── MOE_Lenet.py          # Training script implementing incremental learning with MoE
├── inference_Lenet.py    # Inference script for evaluating the trained MoE model
└── checkpoints_Lenet/    # Directory containing model checkpoints (moe_model_task1.pt, moe_model_task2.pt, etc.)
```

## How to Run

### Prerequisites

- **Python 3.7+**
- **PyTorch** and **torchvision**
- Additional libraries: NumPy, Pandas, Matplotlib, tqdm, PIL

### Training

1. Ensure the GTSRB dataset is available or allow the script to download it automatically by running the training script:
   ```bash
   python MOE_Lenet.py
   ```
2. The training script will save checkpoints for each task in the `checkpoints_Lenet` directory.

### Inference

1. In `inference_Lenet.py`, verify that the `checkpoint_path` points to the desired checkpoint (e.g., `./checkpoints_Lenet/moe_model_task4.pt`).
2. Run the inference script:
   ```bash
   python inference_Lenet.py
   ```
3. The script will load the saved model and output the test accuracy (approximately 75.84%).

## Context and Future Work

Incremental learning is essential in many real-world applications where models must adapt to new data without undergoing complete retraining. This project demonstrates a practical approach using a modular architecture where new experts are added as new classes are introduced. Future work may include:
- Exploring more advanced memory buffer strategies.
- Improving the router mechanism for better expert selection.
- Applying the approach to more complex or larger-scale datasets.

## Acknowledgements

- **GTSRB Dataset:** This project utilizes the German Traffic Sign Recognition Benchmark for training and evaluation.
- Thanks to the research community for ongoing advancements in incremental learning and mixture-of-experts methodologies.

---

For questions, suggestions, or contributions, please open an issue or submit a pull request in the repository.

