"""
Mini-batch Sampling Strategies for Long-Tailed Image Classification
CIFAR-100-LT Experiments

Author: Siyu Yuan, University of Bristol
Date: February 2026

"""

import os
import sys
import random
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision
import torchvision.transforms as transforms
from collections import Counter
from datetime import datetime


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============================================================
# Global Configuration
# ============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard training config for CIFAR-100-LT (following Cao et al. 2019, Kang et al. 2020)
CONFIG = {
    'epochs': 200,
    'batch_size': 128,           # 128 is standard for CIFAR (was 64)
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 2e-4,
    'warmup_epochs': 5,          # NEW: LR warmup
    'lr_milestones': [160, 180], # NEW: Multi-step schedule (standard)
    'lr_gamma': 0.01,            # Decay factor at milestones
    'grad_clip': 5.0,            # NEW: Gradient clipping
    'num_workers': 2,
    'strategies': ['uniform', 'class_balanced', 'square_root', 'progressive'],
    'imbalance_ratios': [10, 50, 100],
    'seeds': [42, 123, 456],
    'save_dir': './results',
    'figure_dir': './figures',
}


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 2. Dataset Construction
# ============================================================

def create_cifar100_lt(imbalance_ratio, root='./data', seed=42):
    """
    Create long-tailed CIFAR-100 dataset with exponential decay.
    
    Args:
        imbalance_ratio: ratio n_1 / n_K (rho)
        root: data directory
        seed: random seed for reproducible subset selection
    
    Returns:
        train_dataset, test_dataset, class_counts, selected_indices
    """
    # Standard CIFAR-100 normalization
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load original CIFAR-100
    full_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )
    
    targets = np.array(full_train.targets)
    num_classes = 100
    n_max = 500  # Original samples per class
    
    # Exponential decay: n_k = n_1 * mu^(k-1), where mu = rho^(-1/(K-1))
    mu = imbalance_ratio ** (-1.0 / (num_classes - 1))
    class_counts_target = [max(int(n_max * (mu ** k)), 1) for k in range(num_classes)]
    
    # Use a local RNG for reproducible subset selection
    rng = np.random.RandomState(seed)
    
    selected_indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        rng.shuffle(class_indices)
        n_select = min(class_counts_target[class_idx], len(class_indices))
        selected_indices.extend(class_indices[:n_select].tolist())
    
    # Create subset
    train_dataset = Subset(full_train, selected_indices)
    
    # Compute actual class counts
    actual_counts = Counter(targets[i] for i in selected_indices)
    class_counts = [actual_counts.get(i, 0) for i in range(num_classes)]
    
    return train_dataset, test_dataset, class_counts, selected_indices


def get_dataset_info(class_counts, imbalance_ratio):
    """Print dataset statistics."""
    print(f"\n  CIFAR-100-LT (rho={imbalance_ratio})")
    print(f"  Total training samples: {sum(class_counts)}")
    print(f"  Max class size: {max(class_counts)}")
    print(f"  Min class size: {min(class_counts)}")
    print(f"  Median class size: {sorted(class_counts)[50]}")
    
    head = sum(1 for c in class_counts if c > 100)
    medium = sum(1 for c in class_counts if 20 < c <= 100)
    tail = sum(1 for c in class_counts if c <= 20)
    print(f"  Head/Medium/Tail classes: {head}/{medium}/{tail}")
    return head, medium, tail


# ============================================================
# 3. Sampling Strategies
# ============================================================

def get_labels_from_dataset(dataset):
    """Extract labels from a dataset (handles Subset)."""
    if isinstance(dataset, Subset):
        original_targets = np.array(dataset.dataset.targets)
        return [int(original_targets[i]) for i in dataset.indices]
    else:
        return list(dataset.targets)


def create_sampler(strategy, labels, class_counts, epoch=0, total_epochs=200):
    """
    Create a WeightedRandomSampler for the given strategy.
    
    Args:
        strategy: 'uniform', 'class_balanced', 'square_root', 'progressive'
        labels: list of class labels for each sample
        class_counts: list of sample counts per class
        epoch: current epoch (for progressive)
        total_epochs: total training epochs
    
    Returns:
        WeightedRandomSampler or None (for uniform)
    """
    num_classes = len(class_counts)
    n_total = sum(class_counts)
    
    if strategy == 'uniform':
        # Standard uniform instance sampling - use DataLoader shuffle
        return None
    
    elif strategy == 'class_balanced':
        # P(class k) = 1/K for all k
        # Weight for sample i with label y_i: w_i = 1 / (K * n_{y_i})
        weights = []
        for label in labels:
            w = 1.0 / (num_classes * max(class_counts[label], 1))
            weights.append(w)
    
    elif strategy == 'square_root':
        # P(class k) proportional to sqrt(n_k)
        # This is intermediate between uniform and class-balanced
        sqrt_counts = [np.sqrt(max(c, 1)) for c in class_counts]
        Z = sum(sqrt_counts)
        # P(class k) = sqrt(n_k) / Z
        # P(sample i | class k) = 1 / n_k  
        # P(sample i) = P(class k) * P(sample i | class k) = sqrt(n_k) / (Z * n_k)
        weights = []
        for label in labels:
            w = np.sqrt(max(class_counts[label], 1)) / (Z * max(class_counts[label], 1))
            weights.append(w)
    
    elif strategy == 'progressive':
        # Linear interpolation from uniform to class-balanced
        # lambda goes from 0 (uniform) to 1 (class-balanced) over training
        # Using epoch/(total_epochs-1) so lambda reaches exactly 1.0 at last epoch
        if total_epochs > 1:
            lam = min(epoch / (total_epochs - 1), 1.0)
        else:
            lam = 1.0
        
        # Effective class probability:
        # P_eff(k) = (1 - lambda) * (n_k / n) + lambda * (1 / K)
        class_probs = []
        for k in range(num_classes):
            p_uniform = class_counts[k] / n_total
            p_balanced = 1.0 / num_classes
            p_eff = (1.0 - lam) * p_uniform + lam * p_balanced
            class_probs.append(p_eff)
        
        # Weight for sample i: P_eff(y_i) / n_{y_i}
        weights = []
        for label in labels:
            w = class_probs[label] / max(class_counts[label], 1)
            weights.append(w)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    weights_tensor = torch.DoubleTensor(weights)
    return WeightedRandomSampler(
        weights_tensor, 
        num_samples=len(weights_tensor), 
        replacement=True
    )


# ============================================================
# 4. ResNet-32 Model
# ============================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-32."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR (32x32 input)."""
    
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        # Proper weight initialization (He initialization)
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


# ============================================================
# 5. Learning Rate Schedule with Warmup
# ============================================================

def get_lr(epoch, config):
    """
    Multi-step LR with linear warmup.
    
    Warmup: linearly increase from 0 to base_lr over warmup_epochs.
    Multi-step: decay by lr_gamma at each milestone.
    """
    base_lr = config['lr']
    warmup_epochs = config['warmup_epochs']
    
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    
    # Multi-step decay
    lr = base_lr
    for milestone in config['lr_milestones']:
        if epoch >= milestone:
            lr *= config['lr_gamma']
    
    return lr


def adjust_learning_rate(optimizer, epoch, config):
    """Set learning rate for the given epoch."""
    lr = get_lr(epoch, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================
# 6. Training and Evaluation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, config):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, class_counts=None, num_classes=100):
    """
    Evaluate model on test set.
    Returns overall accuracy and per-group (head/medium/tail) accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    overall_acc = 100.0 * correct / total
    avg_loss = total_loss / total
    
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = 100.0 * class_correct[i] / class_total[i]
    
    # Group accuracy based on training class counts
    head_acc = medium_acc = tail_acc = 0.0
    if class_counts is not None:
        head_idx = [i for i in range(num_classes) if class_counts[i] > 100]
        medium_idx = [i for i in range(num_classes) if 20 < class_counts[i] <= 100]
        tail_idx = [i for i in range(num_classes) if class_counts[i] <= 20]
        
        if head_idx:
            head_acc = float(np.mean([per_class_acc[i] for i in head_idx]))
        if medium_idx:
            medium_acc = float(np.mean([per_class_acc[i] for i in medium_idx]))
        if tail_idx:
            tail_acc = float(np.mean([per_class_acc[i] for i in tail_idx]))
    
    return {
        'overall_acc': overall_acc,
        'head_acc': head_acc,
        'medium_acc': medium_acc,
        'tail_acc': tail_acc,
        'per_class_acc': per_class_acc.tolist(),
        'avg_loss': avg_loss,
    }


# ============================================================
# 7. Main Experiment Runner
# ============================================================

def run_single_experiment(strategy, imbalance_ratio, seed, config):
    """
    Run a single experiment (one strategy, one rho, one seed).
    
    Returns a comprehensive results dictionary.
    """
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy} | rho: {imbalance_ratio} | Seed: {seed}")
    print(f"{'='*60}")
    
    # ---- Dataset ----
    train_dataset, test_dataset, class_counts, _ = create_cifar100_lt(
        imbalance_ratio, seed=seed
    )
    get_dataset_info(class_counts, imbalance_ratio)
    
    # Pre-extract labels (only needs to be done once)
    labels = get_labels_from_dataset(train_dataset)
    
    # Test loader (fixed)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    # ---- Model ----
    set_seed(seed)  # Re-seed to ensure same model init across strategies
    model = ResNet32(num_classes=100).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['lr'], 
        momentum=config['momentum'], 
        weight_decay=config['weight_decay']
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    # ---- Results storage ----
    history = {
        'strategy': strategy,
        'imbalance_ratio': imbalance_ratio,
        'seed': seed,
        'class_counts': class_counts,
        'config': {k: str(v) if not isinstance(v, (int, float, str, list)) else v 
                   for k, v in config.items()},
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_loss': [],
        'head_acc': [],
        'medium_acc': [],
        'tail_acc': [],
        'lr_history': [],
    }
    
    best_acc = 0.0
    best_model_state = None
    
    # ---- Training loop ----
    epochs = config['epochs']
    pbar = tqdm(range(epochs), desc=f'{strategy}|rho={imbalance_ratio}|s={seed}')
    
    for epoch in pbar:
        # Adjust learning rate (warmup + multi-step)
        lr = adjust_learning_rate(optimizer, epoch, config)
        history['lr_history'].append(lr)
        
        # Create sampler for this epoch
        sampler = create_sampler(strategy, labels, class_counts, epoch, epochs)
        
        if sampler is None:
            train_loader = DataLoader(
                train_dataset, batch_size=config['batch_size'], 
                shuffle=True, num_workers=config['num_workers'],
                pin_memory=True, drop_last=True, 
                worker_init_fn=worker_init_fn
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=config['batch_size'],
                sampler=sampler, num_workers=config['num_workers'],
                pin_memory=True, drop_last=True,
                worker_init_fn=worker_init_fn
            )
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config
        )
        
        # Evaluate
        eval_results = evaluate(model, test_loader, criterion, class_counts)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(eval_results['overall_acc'])
        history['test_loss'].append(eval_results['avg_loss'])
        history['head_acc'].append(eval_results['head_acc'])
        history['medium_acc'].append(eval_results['medium_acc'])
        history['tail_acc'].append(eval_results['tail_acc'])
        
        # Track best
        if eval_results['overall_acc'] > best_acc:
            best_acc = eval_results['overall_acc']
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Progress bar
        pbar.set_postfix({
            'lr': f'{lr:.4f}',
            'loss': f'{train_loss:.3f}',
            'acc': f'{eval_results["overall_acc"]:.1f}',
            'H/M/T': f'{eval_results["head_acc"]:.0f}/{eval_results["medium_acc"]:.0f}/{eval_results["tail_acc"]:.0f}'
        })
    
    # ---- Final evaluation with best model ----
    model.load_state_dict(best_model_state)
    final_eval = evaluate(model, test_loader, criterion, class_counts)
    
    history['best_acc'] = best_acc
    history['final_overall_acc'] = final_eval['overall_acc']
    history['final_head_acc'] = final_eval['head_acc']
    history['final_medium_acc'] = final_eval['medium_acc']
    history['final_tail_acc'] = final_eval['tail_acc']
    history['final_per_class_acc'] = final_eval['per_class_acc']
    
    print(f"\n  Best Test Acc: {best_acc:.2f}%")
    print(f"  Final (best ckpt): Overall={final_eval['overall_acc']:.2f}%, "
          f"Head={final_eval['head_acc']:.2f}%, "
          f"Medium={final_eval['medium_acc']:.2f}%, "
          f"Tail={final_eval['tail_acc']:.2f}%")
    
    # Save results
    os.makedirs(config['save_dir'], exist_ok=True)
    filename = os.path.join(
        config['save_dir'], 
        f"results_{strategy}_rho{imbalance_ratio}_seed{seed}.json"
    )
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved: {filename}")
    
    return history


# ============================================================
# 8. Full Experiment Suite
# ============================================================

def run_all_experiments(config):
    """Run all strategy x rho x seed combinations."""
    all_results = {}
    
    total_runs = (len(config['strategies']) * len(config['imbalance_ratios']) 
                  * len(config['seeds']))
    run_idx = 0
    
    for rho in config['imbalance_ratios']:
        for strategy in config['strategies']:
            for seed in config['seeds']:
                run_idx += 1
                print(f"\n{'#'*60}")
                print(f"  RUN {run_idx}/{total_runs}")
                print(f"{'#'*60}")
                
                t0 = time.time()
                result = run_single_experiment(strategy, rho, seed, config)
                elapsed = time.time() - t0
                print(f"  Time: {elapsed/60:.1f} min")
                
                key = (strategy, rho, seed)
                all_results[key] = result
    
    return all_results


# ============================================================
# 9. Results Aggregation
# ============================================================

def aggregate_results(all_results, config):
    """
    Aggregate results across seeds.
    Returns a nested dict: results[strategy][rho] = {mean, std, ...}
    """
    aggregated = {}
    
    for strategy in config['strategies']:
        aggregated[strategy] = {}
        for rho in config['imbalance_ratios']:
            seed_results = []
            for seed in config['seeds']:
                key = (strategy, rho, seed)
                if key in all_results:
                    seed_results.append(all_results[key])
            
            if not seed_results:
                continue
            
            # Aggregate final accuracies
            best_accs = [r['best_acc'] for r in seed_results]
            final_accs = [r['final_overall_acc'] for r in seed_results]
            head_accs = [r['final_head_acc'] for r in seed_results]
            medium_accs = [r['final_medium_acc'] for r in seed_results]
            tail_accs = [r['final_tail_acc'] for r in seed_results]
            
            # Aggregate curves
            test_acc_curves = [r['test_acc'] for r in seed_results]
            train_loss_curves = [r['train_loss'] for r in seed_results]
            head_curves = [r['head_acc'] for r in seed_results]
            medium_curves = [r['medium_acc'] for r in seed_results]
            tail_curves = [r['tail_acc'] for r in seed_results]
            
            aggregated[strategy][rho] = {
                'best_acc_mean': np.mean(best_accs),
                'best_acc_std': np.std(best_accs),
                'final_acc_mean': np.mean(final_accs),
                'final_acc_std': np.std(final_accs),
                'head_acc_mean': np.mean(head_accs),
                'head_acc_std': np.std(head_accs),
                'medium_acc_mean': np.mean(medium_accs),
                'medium_acc_std': np.std(medium_accs),
                'tail_acc_mean': np.mean(tail_accs),
                'tail_acc_std': np.std(tail_accs),
                'test_acc_curves': np.array(test_acc_curves),
                'train_loss_curves': np.array(train_loss_curves),
                'head_curves': np.array(head_curves),
                'medium_curves': np.array(medium_curves),
                'tail_curves': np.array(tail_curves),
                'class_counts': seed_results[0]['class_counts'],
            }
    
    return aggregated


# ============================================================
# 10. LaTeX Table Generation
# ============================================================

def generate_latex_table(aggregated, config):
    """Generate LaTeX table for the paper (Table 2: Overall Test Accuracy)."""
    
    strategy_names = {
        'uniform': 'Uniform',
        'class_balanced': 'Class-Balanced',
        'square_root': 'Square-Root',
        'progressive': 'Progressive',
    }
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall test accuracy (\%) on CIFAR-100-LT across different "
                 r"sampling strategies and imbalance ratios. Results are reported as "
                 r"mean $\pm$ std over 3 random seeds.}")
    lines.append(r"\label{tab:overall_results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Strategy} & $\boldsymbol{\rho = 10}$ & "
                 r"$\boldsymbol{\rho = 50}$ & $\boldsymbol{\rho = 100}$ \\")
    lines.append(r"\midrule")
    
    for strategy in config['strategies']:
        name = strategy_names[strategy]
        row = f"{name}"
        for rho in config['imbalance_ratios']:
            if rho in aggregated[strategy]:
                data = aggregated[strategy][rho]
                row += f" & ${data['best_acc_mean']:.1f} \\pm {data['best_acc_std']:.1f}$"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_str = "\n".join(lines)
    
    # Save
    os.makedirs(config['save_dir'], exist_ok=True)
    filepath = os.path.join(config['save_dir'], 'results_table.tex')
    with open(filepath, 'w') as f:
        f.write(table_str)
    
    print("\nLaTeX Table:")
    print("=" * 60)
    print(table_str)
    
    return table_str


def generate_head_medium_tail_table(aggregated, config):
    """Generate LaTeX table for head/medium/tail breakdown."""
    
    strategy_names = {
        'uniform': 'Uniform',
        'class_balanced': 'Class-Balanced',
        'square_root': 'Square-Root',
        'progressive': 'Progressive',
    }
    
    for rho in config['imbalance_ratios']:
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{Head/Medium/Tail accuracy (\\%) on CIFAR-100-LT "
                     f"with $\\rho = {rho}$.}}")
        lines.append(f"\\label{{tab:hmt_rho{rho}}}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Strategy} & \textbf{Overall} & \textbf{Head} & "
                     r"\textbf{Medium} & \textbf{Tail} \\")
        lines.append(r"\midrule")
        
        for strategy in config['strategies']:
            name = strategy_names[strategy]
            if rho in aggregated[strategy]:
                d = aggregated[strategy][rho]
                lines.append(
                    f"{name} & ${d['best_acc_mean']:.1f}$ & "
                    f"${d['head_acc_mean']:.1f}$ & "
                    f"${d['medium_acc_mean']:.1f}$ & "
                    f"${d['tail_acc_mean']:.1f}$ \\\\"
                )
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        filepath = os.path.join(config['save_dir'], f'hmt_table_rho{rho}.tex')
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))


# ============================================================
# 11. Figure Generation
# ============================================================

def plot_figures(aggregated, config):
    """Generate all figures for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(config['figure_dir'], exist_ok=True)
    
    # Color scheme
    colors = {
        'uniform': '#1f77b4',       # Blue
        'class_balanced': '#ff7f0e', # Orange
        'square_root': '#2ca02c',    # Green
        'progressive': '#d62728',    # Red
    }
    labels = {
        'uniform': 'Uniform',
        'class_balanced': 'Class-Balanced',
        'square_root': 'Square-Root',
        'progressive': 'Progressive',
    }
    
    # ---- Figure 1: Overall Accuracy Bar Chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(config['imbalance_ratios']))
    width = 0.2
    
    for i, strategy in enumerate(config['strategies']):
        means = []
        stds = []
        for rho in config['imbalance_ratios']:
            if rho in aggregated[strategy]:
                means.append(aggregated[strategy][rho]['best_acc_mean'])
                stds.append(aggregated[strategy][rho]['best_acc_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i * width, means, width, yerr=stds, 
               label=labels[strategy], color=colors[strategy],
               capsize=3, alpha=0.85)
    
    ax.set_xlabel('Imbalance Ratio (ρ)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f'ρ = {r}' for r in config['imbalance_ratios']])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['figure_dir'], 'overall_accuracy_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config['figure_dir'], 'overall_accuracy_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Figure 2: Convergence Curves (one per rho) ----
    for rho in config['imbalance_ratios']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for strategy in config['strategies']:
            if rho not in aggregated[strategy]:
                continue
            data = aggregated[strategy][rho]
            
            # Test accuracy curve
            curves = data['test_acc_curves']
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            epochs = np.arange(1, len(mean_curve) + 1)
            
            axes[0].plot(epochs, mean_curve, label=labels[strategy], 
                        color=colors[strategy], linewidth=2)
            axes[0].fill_between(epochs, mean_curve - std_curve, 
                                mean_curve + std_curve,
                                color=colors[strategy], alpha=0.15)
            
            # Training loss curve
            loss_curves = data['train_loss_curves']
            mean_loss = np.mean(loss_curves, axis=0)
            axes[1].plot(epochs, mean_loss, label=labels[strategy],
                        color=colors[strategy], linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Test Accuracy (%)', fontsize=11)
        axes[0].set_title(f'Test Accuracy During Training (ρ = {rho})', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Training Loss', fontsize=11)
        axes[1].set_title(f'Training Loss During Training (ρ = {rho})', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'convergence_curves_rho{rho}.pdf'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'convergence_curves_rho{rho}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Figure 3: Head/Medium/Tail Breakdown (grouped bar, one per rho) ----
    for rho in config['imbalance_ratios']:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(3)  # Head, Medium, Tail
        width = 0.2
        
        for i, strategy in enumerate(config['strategies']):
            if rho not in aggregated[strategy]:
                continue
            d = aggregated[strategy][rho]
            means = [d['head_acc_mean'], d['medium_acc_mean'], d['tail_acc_mean']]
            stds = [d['head_acc_std'], d['medium_acc_std'], d['tail_acc_std']]
            
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=labels[strategy], color=colors[strategy],
                   capsize=3, alpha=0.85)
        
        ax.set_xlabel('Class Group', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Head/Medium/Tail Accuracy (ρ = {rho})', fontsize=14)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(['Head\n(Many-shot)', 'Medium\n(Medium-shot)', 
                           'Tail\n(Few-shot)'])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'head_medium_tail_rho{rho}.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'head_medium_tail_rho{rho}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # ---- Figure 4: Effect of Imbalance Ratio ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    group_labels = ['Head Accuracy', 'Medium Accuracy', 'Tail Accuracy']
    group_keys = ['head_acc_mean', 'medium_acc_mean', 'tail_acc_mean']
    
    for ax, group_label, key in zip(axes, group_labels, group_keys):
        for strategy in config['strategies']:
            rhos = []
            accs = []
            for rho in config['imbalance_ratios']:
                if rho in aggregated[strategy]:
                    rhos.append(rho)
                    accs.append(aggregated[strategy][rho][key])
            
            ax.plot(rhos, accs, 'o-', label=labels[strategy], 
                   color=colors[strategy], linewidth=2, markersize=8)
        
        ax.set_xlabel('Imbalance Ratio (ρ)', fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.set_title(group_label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(config['imbalance_ratios'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['figure_dir'], 'imbalance_effect.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(config['figure_dir'], 'imbalance_effect.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Figure 5: Head/Medium/Tail accuracy curves during training ----
    for rho in config['imbalance_ratios']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Head Classes', 'Medium Classes', 'Tail Classes']
        curve_keys = ['head_curves', 'medium_curves', 'tail_curves']
        
        for ax, title, curve_key in zip(axes, titles, curve_keys):
            for strategy in config['strategies']:
                if rho not in aggregated[strategy]:
                    continue
                data = aggregated[strategy][rho]
                curves = data[curve_key]
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                epochs = np.arange(1, len(mean_curve) + 1)
                
                ax.plot(epochs, mean_curve, label=labels[strategy],
                       color=colors[strategy], linewidth=2)
                ax.fill_between(epochs, mean_curve - std_curve,
                              mean_curve + std_curve,
                              color=colors[strategy], alpha=0.15)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'{title} (ρ = {rho})', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'group_curves_rho{rho}.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(config['figure_dir'], 
                    f'group_curves_rho{rho}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll figures saved to {config['figure_dir']}/")


# ============================================================
# 12. Load Existing Results (if resuming)
# ============================================================

def load_existing_results(config):
    """Load any previously saved results to allow resuming."""
    all_results = {}
    save_dir = config['save_dir']
    
    if not os.path.exists(save_dir):
        return all_results
    
    for fname in os.listdir(save_dir):
        if fname.startswith('results_') and fname.endswith('.json'):
            filepath = os.path.join(save_dir, fname)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                key = (data['strategy'], data['imbalance_ratio'], data['seed'])
                all_results[key] = data
                print(f"  Loaded: {fname}")
            except Exception as e:
                print(f"  Warning: Could not load {fname}: {e}")
    
    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  CIFAR-100-LT Sampling Strategy Experiments")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    config = CONFIG
    
    # Check for existing results
    print("\nChecking for existing results...")
    all_results = load_existing_results(config)
    print(f"  Found {len(all_results)} existing results")
    
    # Run missing experiments
    total_needed = (len(config['strategies']) * len(config['imbalance_ratios']) 
                   * len(config['seeds']))
    
    for rho in config['imbalance_ratios']:
        for strategy in config['strategies']:
            for seed in config['seeds']:
                key = (strategy, rho, seed)
                if key in all_results:
                    print(f"\n  Skipping {strategy}/rho={rho}/seed={seed} (already done)")
                    continue
                
                result = run_single_experiment(strategy, rho, seed, config)
                all_results[key] = result
    
    # Aggregate and generate outputs
    print("\n" + "=" * 60)
    print("  Generating tables and figures...")
    print("=" * 60)
    
    aggregated = aggregate_results(all_results, config)
    generate_latex_table(aggregated, config)
    generate_head_medium_tail_table(aggregated, config)
    plot_figures(aggregated, config)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    
    strategy_names = {
        'uniform': 'Uniform',
        'class_balanced': 'Class-Balanced',
        'square_root': 'Square-Root',
        'progressive': 'Progressive',
    }
    
    for rho in config['imbalance_ratios']:
        print(f"\n  rho = {rho}:")
        print(f"  {'Strategy':<18} {'Overall':>10} {'Head':>10} {'Medium':>10} {'Tail':>10}")
        print(f"  {'-'*58}")
        for strategy in config['strategies']:
            if rho in aggregated[strategy]:
                d = aggregated[strategy][rho]
                print(f"  {strategy_names[strategy]:<18} "
                      f"{d['best_acc_mean']:>7.1f}±{d['best_acc_std']:.1f} "
                      f"{d['head_acc_mean']:>7.1f}±{d['head_acc_std']:.1f} "
                      f"{d['medium_acc_mean']:>7.1f}±{d['medium_acc_std']:.1f} "
                      f"{d['tail_acc_mean']:>7.1f}±{d['tail_acc_std']:.1f}")
    
    print(f"\n  Results saved to: {config['save_dir']}/")
    print(f"  Figures saved to: {config['figure_dir']}/")
    print("\nDone!")
