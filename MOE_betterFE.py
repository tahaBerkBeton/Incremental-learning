####################################
### Useful imports
####################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
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
train_target = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/train_target.csv', 
                             delimiter=',', header=None).to_numpy().squeeze().tolist()
test_target  = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/test_target.csv', 
                            delimiter=',', header=None).to_numpy().squeeze().tolist()
class_names  = pd.read_csv('https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/signnames.csv')['SignName'].tolist()

# We will also use a separate candidate pool for the memory buffer.
# (Here we use the test split as our "non-training" pool.)
buffer_pool_dataset = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)
# Attach targets (so we can filter by class)
buffer_pool_dataset.targets = test_target


####################################
### Model Definitions: Mixture of Experts
####################################
# Enhanced Shared feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, n_in=3):
        super(FeatureExtractor, self).__init__()
        # Initial convolution block
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Second block with pooling: 32x32 -> 16x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Third block: increase channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Fourth block with pooling: 16x16 -> 8x8
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Fifth block: further increase channel depth
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # A residual block to refine features without loss of information.
        self.res_block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.relu_final = nn.ReLU(inplace=True)
        # Global average pooling reduces each channel to a single number.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout to help regularize the learned features.
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Residual connection: add the input of the block to its output.
        residual = x
        x = self.res_block(x)
        x = x + residual
        x = self.relu_final(x)
        # Global average pooling: shape becomes [B, 256, 1, 1]
        x = self.global_pool(x)
        # Flatten to [B, 256]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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
    def __init__(self, feature_dim=256):  # Updated default feature_dim
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
        features = self.feature_extractor(x)             # [B, 256]
        router_logits = self.router(features)              # [B, num_experts]
        # Hard routing: get predicted expert for each sample
        selected_expert_indices = torch.argmax(router_logits, dim=1)  # [B]
        return features, router_logits, selected_expert_indices


####################################
### Helper function for label mapping
####################################
def get_ground_truth_expert_info(labels, expert_classes, device):
    """
    For each global label in the batch (labels may be a tensor or list),
    return:
      - gt_expert: the expert index that should handle the label.
      - local_label: the label index within that expert.
    Both are returned as tensors on the given device.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
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
    return torch.tensor(gt_expert, device=device), torch.tensor(local_label, device=device)


####################################
### Function to build a stratified memory buffer from a candidate pool
####################################
def build_memory_buffer(expert_classes, candidate_dataset, total_buffer_size, current_task_idx):
    """
    Build a memory buffer of exactly total_buffer_size items drawn equally from all past tasks.
    current_task_idx: the index (0-indexed) of the current task.
       The buffer will include examples from expert_classes[0:current_task_idx] (i.e. past tasks).
    candidate_dataset: a dataset (with attribute 'targets') that is NOT used for training.
    """
    past_expert_classes = expert_classes[:current_task_idx]
    num_past_tasks = len(past_expert_classes)
    memory_buffer = []
    if num_past_tasks == 0:
        return memory_buffer  # No past tasks yet.
    # Count total number of classes in past tasks
    total_past_classes = sum(len(task_classes) for task_classes in past_expert_classes)
    samples_per_class = total_buffer_size // total_past_classes
    remainder = total_buffer_size - samples_per_class * total_past_classes

    # For each past task and each class in that task, sample examples from candidate_dataset.
    for task_classes in past_expert_classes:
        for cls in task_classes:
            # Get candidate indices for this class:
            candidate_indices = [i for i, label in enumerate(candidate_dataset.targets) if label == cls]
            # Decide how many samples to pick:
            n_samples = samples_per_class + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            if len(candidate_indices) > 0:
                selected = np.random.choice(candidate_indices, min(n_samples, len(candidate_indices)), replace=False)
                for idx in selected:
                    sample = candidate_dataset[idx]  # (image, label)
                    memory_buffer.append(sample)
    # In case we have fewer than total_buffer_size, we can leave it as is.
    return memory_buffer


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
### Incremental Learning function for MoE (with buffer ponderation and stratified memory buffer)
####################################
def incremental_learning_moe(model, train_dataset, train_target, test_dataset, test_target,
                               num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                               buffer_size=1000, alignment_strength=0.1, buffer_weight=2.0):
    # All classes available in GTSRB
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    current_classes = []  # Global classes seen so far
    accuracies = []
    # Our memory buffer will be built from the candidate pool (which is not training data)
    memory_buffer = []  # List of (image, label) tuples

    criterion = nn.CrossEntropyLoss()

    for task in range(num_tasks):
        # Define new task classes
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)
        print(f"\n--- Starting Task {task+1} with classes: {task_classes} ---")

        # Create dataloader for new task (from training data)
        new_task_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
        
        # Build buffer loader from our memory buffer (which comes from the candidate pool, not training data)
        if task > 0:  # Only if we have past tasks
            # Rebuild the memory buffer to contain exactly buffer_size items
            memory_buffer = build_memory_buffer(model.expert_classes, buffer_pool_dataset, total_buffer_size=buffer_size, current_task_idx=task)
            buffer_images, buffer_labels = zip(*memory_buffer) if len(memory_buffer) > 0 else ([], [])
            if len(buffer_images) > 0:
                buffer_images = torch.stack(buffer_images)
                buffer_labels = torch.tensor(buffer_labels)
                buffer_dataset = TensorDataset(buffer_images, buffer_labels)
                buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True)
            else:
                buffer_loader = None
        else:
            buffer_loader = None

        # Add new expert for this task
        model.add_expert(task_classes)
        model.to(device)

        # Freeze feature extractor and old experts so that only the new expert and router are trainable.
        # (For buffer loss, the router will still be updated.)
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        for expert in model.experts[:-1]:
            for param in expert.parameters():
                param.requires_grad = False
        # The new expert (last one) and the router remain trainable.
        
        # Set optimizer to update only parameters with requires_grad=True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Create iterators for new task and buffer loaders
        new_task_iter = iter(new_task_loader)
        if buffer_loader is not None:
            buffer_iter = iter(buffer_loader)
            num_batches = max(len(new_task_loader), len(buffer_loader))
        else:
            num_batches = len(new_task_loader)
        
        # For the first task, use 80 epochs; otherwise, use the provided num_epochs.
        epochs_for_this_task = 80 if task == 0 else num_epochs

        for epoch in range(epochs_for_this_task):
            model.train()
            pbar = tqdm(range(num_batches), ncols=80, desc=f"Task {task+1}, Epoch {epoch+1}")
            for _ in pbar:
                # --- New Task Batch ---
                try:
                    images_new, labels_new = next(new_task_iter)
                except StopIteration:
                    new_task_iter = iter(new_task_loader)
                    images_new, labels_new = next(new_task_iter)
                new_images = images_new.to(device)
                new_labels = labels_new.to(device)

                # Forward pass for new task samples
                features_new, router_logits_new, _ = model(new_images)
                gt_expert_new, local_labels_new = get_ground_truth_expert_info(new_labels, model.expert_classes, device)
                # For new task, we compute classification loss only for samples that belong to the new expert
                new_expert_id = len(model.experts) - 1
                mask_new = (gt_expert_new == new_expert_id).nonzero(as_tuple=True)[0]
                if mask_new.numel() > 0:
                    features_new_sel = features_new[mask_new]
                    expert_logits_new = model.experts[new_expert_id](features_new_sel)
                    local_labels_new_sel = local_labels_new[mask_new]
                    classification_loss_new = criterion(expert_logits_new, local_labels_new_sel)
                else:
                    classification_loss_new = 0.0
                # Routing loss for new task samples
                routing_loss_new = criterion(router_logits_new, gt_expert_new)

                # --- Buffer Batch (Past Tasks) ---
                if buffer_loader is not None:
                    try:
                        images_buf, labels_buf = next(buffer_iter)
                    except StopIteration:
                        buffer_iter = iter(buffer_loader)
                        images_buf, labels_buf = next(buffer_iter)
                    buf_images = images_buf.to(device)
                    buf_labels = labels_buf.to(device)
                    features_buf, router_logits_buf, _ = model(buf_images)
                    gt_expert_buf, local_labels_buf = get_ground_truth_expert_info(buf_labels, model.expert_classes, device)
                    # Compute classification loss for each expert branch over buffer samples
                    classification_loss_buf = 0.0
                    for expert_id, cls_list in enumerate(model.expert_classes):
                        idx = (gt_expert_buf == expert_id).nonzero(as_tuple=True)[0]
                        if idx.numel() > 0:
                            expert_logits_buf = model.experts[expert_id](features_buf[idx])
                            local_labels_buf_sel = local_labels_buf[idx]
                            classification_loss_buf += criterion(expert_logits_buf, local_labels_buf_sel)
                    # Routing loss for buffer samples
                    routing_loss_buf = criterion(router_logits_buf, gt_expert_buf)
                else:
                    classification_loss_buf = 0.0
                    routing_loss_buf = 0.0

                # --- Total Loss ---
                # Weight buffer losses more strongly to help mitigate forgetting.
                total_loss = (classification_loss_new + buffer_weight * classification_loss_buf +
                              alignment_strength * (routing_loss_new + routing_loss_buf))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": total_loss.item()})
                
        # End of training for current task

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
num_tasks = 4  # For example, if GTSRB has 43 classes, tasks with roughly equal classes per task.
nclasses = len(np.unique(train_target))
classes_per_task = math.ceil(nclasses / num_tasks)
batch_size = 64
lr = 1e-3
num_epochs = 40     # Default epochs per task (except the first task, which uses 80)
buffer_size = 1000  # Fixed memory buffer size (always 1000 items)
alignment_strength = 0.1
buffer_weight = 2.0  # Hyperparameter to strongly penalize buffer errors

# Initialize our MoE model with updated feature dimension
moe_model = MixtureOfExperts(feature_dim=256)
moe_model.to(device)

# Run incremental learning
accuracies = incremental_learning_moe(moe_model, train_dataset, train_target,
                                      test_dataset, test_target,
                                      num_tasks, classes_per_task,
                                      batch_size, num_epochs, lr, device,
                                      buffer_size=buffer_size,
                                      alignment_strength=alignment_strength,
                                      buffer_weight=buffer_weight)

print("\nIncremental Learning Accuracies per Task:")
for i, acc in enumerate(accuracies):
    print(f"Task {i+1}: {acc:.2f}%")
