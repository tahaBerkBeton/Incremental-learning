####################################
### Useful imports
####################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import torchvision.models as models
from torchvision.transforms import v2

import numpy as np
import random
import time, os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
import copy

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
### Data loaders & Transforms
####################################
# We use torchvision’s GTSRB dataset
transform_train = v2.Compose([
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
transform_test = v2.Compose([
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def get_dataset(root_dir, transform, train=True):
    dataset = datasets.GTSRB(root=root_dir, split='train' if train else 'test', download=True, transform=transform)
    target = [data[1] for data in dataset]
    return dataset, target

def create_dataloader(dataset, targets, selected_classes, batch_size, shuffle):
    indices = [i for i, label in enumerate(targets) if label in selected_classes]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define dataset paths
root_dir = './data'
train_dataset = datasets.GTSRB(root=root_dir, split='train', download=True, transform=transform_train)
test_dataset = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)

# Load target lists and class names (from CSVs)
train_target = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/train_target.csv', delimiter=',', header=None).to_numpy().squeeze().tolist()
test_target  = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/test_target.csv', delimiter=',', header=None).to_numpy().squeeze().tolist()
class_names  = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/signnames.csv')['SignName'].tolist()

####################################
### Model Definitions: Mixture of Experts
####################################
# Shared feature extractor (using conv layers from SimpleCNN)
class FeatureExtractor(nn.Module):
    def __init__(self, n_in=3):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))      # [B, 32, 32, 32]
        x = F.max_pool2d(x, kernel_size=2)     # [B, 32, 16, 16]
        x = F.leaky_relu(self.conv2(x))        # [B, 64, 16, 16]
        x = F.max_pool2d(x, kernel_size=2)     # [B, 64, 8, 8]
        x = F.leaky_relu(self.conv3(x))        # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)              # Flatten to [B, 4096]
        return x

# Expert: a small MLP for a subset of classes
class Expert(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Mixture of Experts Module
class MixtureOfExperts(nn.Module):
    def __init__(self, feature_dim=4096):
        super(MixtureOfExperts, self).__init__()
        self.feature_extractor = FeatureExtractor(n_in=3)
        self.feature_dim = feature_dim
        self.router = None  # Will be initialized when first expert is added
        self.experts = nn.ModuleList()
        self.expert_classes = []  # List of lists: each expert's assigned global classes

    def add_expert(self, new_class_indices):
        """Add a new expert for the given list of global class indices."""
        num_new_classes = len(new_class_indices)
        new_expert = Expert(self.feature_dim, num_new_classes)
        self.experts.append(new_expert)
        self.expert_classes.append(new_class_indices)

        # Update the router: create a new FC layer with output dim = number of experts.
        new_num_experts = len(self.experts)
        old_router = self.router
        new_router = nn.Linear(self.feature_dim, new_num_experts)
        # If there was an old router, copy its weights for the existing experts.
        if old_router is not None:
            with torch.no_grad():
                new_router.weight[:old_router.out_features] = old_router.weight
                new_router.bias[:old_router.out_features] = old_router.bias
        self.router = new_router

    def forward(self, x):
        # Compute shared features and router logits.
        features = self.feature_extractor(x)             # [B, 4096]
        router_logits = self.router(features)              # [B, num_experts]
        # Hard routing: get predicted expert for each sample
        selected_expert_indices = torch.argmax(router_logits, dim=1)  # [B]
        # For evaluation we need to get expert outputs per sample.
        # Here, we return features & router_logits; the training loop will force a forward through the new expert if needed.
        return features, router_logits, selected_expert_indices

####################################
### Helper functions for label mapping
####################################
def get_ground_truth_expert_info(labels, expert_classes):
    """
    For each global label in the batch, return:
      - gt_expert: the expert index that should handle the label.
      - local_label: the label index within that expert.
    Both are returned as tensors.
    """
    gt_expert = []
    local_label = []
    for lbl in labels:
        found = False
        for expert_id, cls_list in enumerate(expert_classes):
            if lbl in cls_list:
                gt_expert.append(expert_id)
                local_label.append(cls_list.index(lbl))
                found = True
                break
        if not found:
            # Should not happen if the expert mapping is correct.
            gt_expert.append(-1)
            local_label.append(-1)
    return torch.tensor(gt_expert, device=labels.device), torch.tensor(local_label, device=labels.device)

####################################
### Evaluation function for MoE
####################################
def evaluate_moe(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, ncols=80, desc="Eval"):
            images, labels = images.to(device), labels.to(device)
            features = model.feature_extractor(images)
            router_logits = model.router(features)
            selected_expert_indices = torch.argmax(router_logits, dim=1)
            # Total number of global classes so far
            total_classes = sum([len(cls_list) for cls_list in model.expert_classes])
            batch_logits = torch.full((images.size(0), total_classes), -1e9, device=device)
            # For each expert, process samples routed to it
            for expert_id, cls_list in enumerate(model.expert_classes):
                idx = (selected_expert_indices == expert_id).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_out = model.experts[expert_id](features[idx])
                    # Place the expert's output in the correct positions of the global logit vector.
                    for i, sample_idx in enumerate(idx):
                        for j, global_class in enumerate(cls_list):
                            batch_logits[sample_idx, global_class] = expert_out[i, j]
            pred = torch.argmax(batch_logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

####################################
### Incremental Learning function for MoE
####################################
def incremental_learning_moe(model, train_dataset, train_target, test_dataset, test_target,
                               num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                               buffer_size=500, alignment_strength=0.1):
    # All classes available in GTSRB
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    current_classes = []  # Global classes seen so far
    accuracies = []
    memory_buffer = []   # List of (image, label) pairs for past tasks

    criterion = nn.CrossEntropyLoss()

    for task in range(num_tasks):
        # Define new task classes
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)
        print(f"\n--- Starting Task {task+1} with classes: {task_classes} ---")

        # Create dataloader for new task
        new_task_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
        # Create dataloader for memory buffer if not empty
        if len(memory_buffer) > 0:
            buffer_images, buffer_labels = zip(*memory_buffer)
            buffer_images = torch.stack(buffer_images)
            buffer_labels = torch.tensor(buffer_labels)
            buffer_dataset = torch.utils.data.TensorDataset(buffer_images, buffer_labels)
            buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True)
        else:
            buffer_loader = None

        # Add new expert for this task
        model.add_expert(task_classes)
        model.to(device)

        # Freeze feature extractor and old experts
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        for expert in model.experts[:-1]:
            for param in expert.parameters():
                param.requires_grad = False
        # The new expert (last one) and the router remain trainable.
        
        # Set optimizer to update only parameters with requires_grad=True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Combine new task loader and memory buffer loader
        # For simplicity, we iterate over new task loader and, if exists, one batch from buffer loader per iteration.
        new_task_iter = iter(new_task_loader)
        buffer_iter = iter(buffer_loader) if buffer_loader is not None else None

        num_batches = max(len(new_task_loader), len(buffer_loader) if buffer_loader is not None else 0)
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(range(num_batches), ncols=80, desc=f"Task {task+1}, Epoch {epoch+1}")
            for _ in pbar:
                # Get a batch from new task loader
                try:
                    images_new, labels_new = next(new_task_iter)
                except StopIteration:
                    new_task_iter = iter(new_task_loader)
                    images_new, labels_new = next(new_task_iter)
                # If memory buffer exists, get a batch from it; else use empty tensors.
                if buffer_iter is not None:
                    try:
                        images_buf, labels_buf = next(buffer_iter)
                    except StopIteration:
                        buffer_iter = iter(buffer_loader)
                        images_buf, labels_buf = next(buffer_iter)
                    # Concatenate new task and buffer data
                    images = torch.cat([images_new, images_buf], dim=0).to(device)
                    labels = torch.cat([labels_new, labels_buf], dim=0).to(device)
                else:
                    images = images_new.to(device)
                    labels = labels_new.to(device)

                # Forward pass through feature extractor and router
                features, router_logits, _ = model(images)
                # Determine ground truth expert and local label for each sample
                gt_expert, local_labels = get_ground_truth_expert_info(labels.cpu().tolist(), model.expert_classes)
                gt_expert = gt_expert.to(device)
                local_labels = local_labels.to(device)
                # Routing loss: encourage router to choose correct expert for each sample.
                routing_loss = criterion(router_logits, gt_expert)

                # Classification loss only for samples that belong to the new expert
                new_expert_id = len(model.experts) - 1
                mask_new = (gt_expert == new_expert_id).nonzero(as_tuple=True)[0]
                if mask_new.numel() > 0:
                    features_new = features[mask_new]
                    # Forward pass through the new expert (regardless of router decision)
                    expert_logits = model.experts[new_expert_id](features_new)
                    local_labels_new = local_labels[mask_new]
                    classification_loss = criterion(expert_logits, local_labels_new)
                else:
                    classification_loss = 0.0

                loss = classification_loss + alignment_strength * routing_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": loss.item()})
                
            # End of epoch

        # After training the new task, update memory buffer.
        # We sample randomly from the new task data (buffer_size per task, here stored overall)
        new_task_images = []
        new_task_labels = []
        for img, lbl in new_task_loader:
            new_task_images.append(img)
            new_task_labels.append(lbl)
        new_task_images = torch.cat(new_task_images, dim=0)
        new_task_labels = torch.tensor(new_task_labels).view(-1)
        # Randomly choose up to (buffer_size // num_tasks) samples from new task
        num_to_sample = min(buffer_size // (task+1), new_task_images.size(0))
        indices = np.random.choice(new_task_images.size(0), num_to_sample, replace=False)
        for idx in indices:
            memory_buffer.append((new_task_images[idx], int(new_task_labels[idx].item())))
        # (Optional) Limit overall memory buffer size if desired.

        # Create test dataloader for all seen classes so far
        test_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        test_acc = evaluate_moe(model, test_loader, device)
        accuracies.append(test_acc)
        print(f"Task {task+1}: Test Accuracy = {test_acc:.2f}%")

    return accuracies

####################################
### Main: Incremental Learning on GTSRB using Mixture-of-Experts
####################################
# Hyperparameters
num_tasks = 5  # For example, if GTSRB has 43 classes, 5 tasks with ~8 classes each (last task may cover fewer)
nclasses = len(np.unique(train_target))
classes_per_task = math.ceil(nclasses / num_tasks)
batch_size = 64
lr = 1e-3
num_epochs = 3      # Adjust epochs per task as needed
buffer_size = 500   # Memory buffer size
alignment_strength = 0.1

# Initialize our MoE model
moe_model = MixtureOfExperts(feature_dim=4096)
moe_model.to(device)

# Run incremental learning
accuracies = incremental_learning_moe(moe_model, train_dataset, train_target,
                                      test_dataset, test_target,
                                      num_tasks, classes_per_task,
                                      batch_size, num_epochs, lr, device,
                                      buffer_size=buffer_size,
                                      alignment_strength=alignment_strength)

print("\nIncremental Learning Accuracies per Task:")
for i, acc in enumerate(accuracies):
    print(f"Task {i+1}: {acc:.2f}%")
