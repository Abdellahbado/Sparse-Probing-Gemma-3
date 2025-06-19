import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
from itertools import combinations
from .config import ProbeConfig


class L0RegularizedLinear(nn.Module):
    """
    Linear layer with L0 regularization for true sparsity.
    Based on "Learning Sparse Neural Networks through L0 Regularization"
    """

    def __init__(self, in_features: int, out_features: int, k: int = None,
                 temperature: float = 2/3, zeta: float = 1.1, gamma: float = -0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k  # Target sparsity level

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # L0 regularization parameters
        self.temperature = temperature
        self.zeta = zeta
        self.gamma = gamma

        # Gate parameters for L0 regularization
        self.log_alpha = nn.Parameter(torch.randn(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.log_alpha, mean=0, std=0.01)

    def sample_z(self, batch_size: int = 1):
        """Sample binary gates"""
        if self.training:
            # Sample from concrete distribution during training
            u = torch.rand(batch_size, self.out_features, self.in_features, device=self.log_alpha.device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature)
            s = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s, 0, 1)
        else:
            # Hard thresholding during inference
            pi = torch.sigmoid(self.log_alpha)
            z = (pi > 0.5).float()

        return z

    def forward(self, x):
        batch_size = x.size(0)
        z = self.sample_z(batch_size)

        # Apply gates to weights
        if self.training:
            masked_weight = self.weight.unsqueeze(0) * z
        else:
            masked_weight = self.weight * z[0]  # Use first sample for inference

        if self.training:
            # Batch matrix multiplication for training
            output = torch.bmm(x.unsqueeze(1), masked_weight.transpose(1, 2)).squeeze(1)
        else:
            output = torch.matmul(x, masked_weight.t())

        return output + self.bias

    def regularization_loss(self):
        """Compute L0 regularization loss"""
        pi = torch.sigmoid(self.log_alpha)
        expected_gates = pi * (self.zeta - self.gamma) + self.gamma
        expected_gates = torch.clamp(expected_gates, 0, 1)
        return torch.sum(expected_gates)

    def get_active_neurons(self):
        """Get indices of active (non-zero) neurons"""
        with torch.no_grad():
            pi = torch.sigmoid(self.log_alpha)
            active_mask = (pi > 0.5)
            active_indices = torch.nonzero(active_mask, as_tuple=True)
            return active_indices

    def enforce_k_sparsity(self):
        """Enforce exactly k active connections"""
        if self.k is not None:
            with torch.no_grad():
                pi = torch.sigmoid(self.log_alpha)
                flat_pi = pi.view(-1)

                # Get top-k values
                _, top_k_indices = torch.topk(flat_pi, self.k)

                # Create mask
                mask = torch.zeros_like(flat_pi)
                mask[top_k_indices] = 1
                mask = mask.view_as(pi)

                # Update log_alpha to enforce sparsity
                self.log_alpha.data = torch.where(
                    mask.bool(),
                    torch.log(torch.tensor(0.99 / 0.01)),  # High probability for selected
                    torch.log(torch.tensor(0.01 / 0.99))   # Low probability for others
                )

class KSparseLinear(nn.Module):
    """
    Simple k-sparse linear layer that selects top-k features.
    Alternative to L0 regularization for more direct sparsity control.
    """

    def __init__(self, in_features: int, out_features: int, k: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Feature selection parameters
        self.feature_importance = nn.Parameter(torch.randn(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)
        nn.init.normal_(self.feature_importance, mean=0, std=0.01)

    def forward(self, x):
        # Select top-k features
        _, top_k_indices = torch.topk(torch.abs(self.feature_importance), self.k)

        # Create mask
        mask = torch.zeros(self.in_features, device=x.device)
        mask[top_k_indices] = 1

        # Apply mask to input
        masked_x = x * mask.unsqueeze(0)

        # Apply linear transformation
        return torch.matmul(masked_x, self.weight.t()) + self.bias

    def get_selected_features(self):
        """Get indices of selected features"""
        with torch.no_grad():
            _, top_k_indices = torch.topk(torch.abs(self.feature_importance), self.k)
            return top_k_indices.cpu().numpy()

class SparseProbe(nn.Module):
    """
    Sparse probe for feature detection in neural network activations.
    """

    def __init__(self, input_dim: int, num_classes: int, k: int = 1,
                 probe_type: str = "k_sparse", **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.k = k
        self.probe_type = probe_type

        if probe_type == "k_sparse":
            self.linear = KSparseLinear(input_dim, num_classes, k)
        elif probe_type == "l0_reg":
            self.linear = L0RegularizedLinear(input_dim, num_classes, k, **kwargs)
        else:
            # Dense probe (normal probing)
            self.linear = nn.Linear(input_dim, num_classes)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

    def get_active_features(self):
        """Get indices of active/selected features"""
        if hasattr(self.linear, 'get_selected_features'):
            return self.linear.get_selected_features()
        elif hasattr(self.linear, 'get_active_neurons'):
            return self.linear.get_active_neurons()
        else:
            # For dense probes, return all features
            return np.arange(self.input_dim)

    def regularization_loss(self):
        """Get regularization loss if applicable"""
        if hasattr(self.linear, 'regularization_loss'):
            return self.linear.regularization_loss()
        return 0.0

class SparseProbeFramework:
    """
    Framework for conducting sparse probing experiments.
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.results = defaultdict(dict)
        self.trained_probes = {}
        self.label_encoders = {}

        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

    def prepare_data(self, activations: torch.Tensor, labels: List[str],
                    layer_name: str) -> Tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
        """
        Prepare activation data and labels for probing.

        Args:
            activations: Activation tensor [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            labels: List of string labels
            layer_name: Name of the layer being probed

        Returns:
            Tuple of (processed_activations, encoded_labels, label_encoder)
        """
        # Convert activations to float32 to avoid dtype mismatches
        activations = activations.to(dtype=torch.float32)

        # Handle different activation shapes
        if len(activations.shape) == 3:
            # Average over sequence length for simplicity
            activations = activations.mean(dim=1)

        # Encode labels
        if layer_name not in self.label_encoders:
            self.label_encoders[layer_name] = LabelEncoder()
            encoded_labels = self.label_encoders[layer_name].fit_transform(labels)
        else:
            encoded_labels = self.label_encoders[layer_name].transform(labels)

        return activations, torch.tensor(encoded_labels, dtype=torch.long), self.label_encoders[layer_name]
    def train_probe(self, X: torch.Tensor, y: torch.Tensor, k: int,
                   probe_type: str = "k_sparse") -> Tuple[SparseProbe, Dict[str, float]]:
        """
        Train a single sparse probe.

        Args:
            X: Input activations [batch_size, features]
            y: Target labels [batch_size]
            k: Sparsity level
            probe_type: Type of probe to use

        Returns:
            Tuple of (trained_probe, training_metrics)
        """
        device = X.device
        num_classes = len(torch.unique(y))
        input_dim = X.shape[1]

        # Create probe
        probe = SparseProbe(input_dim, num_classes, k, probe_type).to(device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(probe.parameters(), lr=self.config.learning_rate)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X.cpu().numpy(), y.cpu().numpy(),
            test_size=0.2, random_state=self.config.random_seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        val_accs = []

        for epoch in range(self.config.num_epochs):
            # Training phase
            probe.train()
            total_loss = 0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i:i+self.config.batch_size]
                batch_y = y_train[i:i+self.config.batch_size]

                optimizer.zero_grad()

                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y)

                # Add regularization loss
                reg_loss = probe.regularization_loss()
                if isinstance(reg_loss, torch.Tensor):
                    loss += self.config.l0_reg * reg_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation phase
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(X_val)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val).float().mean().item()

            train_losses.append(total_loss)
            val_accs.append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        # Enforce k-sparsity if using L0 regularization
        if hasattr(probe.linear, 'enforce_k_sparsity'):
            probe.linear.enforce_k_sparsity()

        training_metrics = {
            'best_val_acc': best_val_acc,
            'final_train_loss': train_losses[-1],
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_accs': val_accs
        }

        return probe, training_metrics

    def evaluate_probe(self, probe: SparseProbe, X_test: torch.Tensor,
                      y_test: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate a trained probe on test data.

        Args:
            probe: Trained probe
            X_test: Test activations
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        probe.eval()
        device = X_test.device

        with torch.no_grad():
            outputs = probe(X_test)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate metrics
            accuracy = (predictions == y_test).float().mean().item()

            # Convert to numpy for sklearn metrics
            y_true = y_test.cpu().numpy()
            y_pred = predictions.cpu().numpy()

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )

            # Get active features
            active_features = probe.get_active_features()

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'active_features': active_features,
                'num_active_features': len(active_features) if isinstance(active_features, np.ndarray) else 0,
                'sparsity_ratio': len(active_features) / probe.input_dim if isinstance(active_features, np.ndarray) else 1.0
            }

        return metrics

    def run_sparse_probing_experiment(self, activations_dict: Dict[str, torch.Tensor],
                                    labels: List[str], task_name: str = "classification") -> Dict[str, Any]:
        """
        Run complete sparse probing experiment across layers and sparsity levels.

        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            labels: List of labels for the classification task
            task_name: Name of the classification task

        Returns:
            Complete experimental results
        """
        print(f"Running sparse probing experiment: {task_name}")
        print(f"Testing k values: {self.config.k_values}")
        print(f"Number of layers: {len(activations_dict)}")

        experiment_results = {
            'task_name': task_name,
            'config': self.config,
            'layer_results': {},
            'summary': {}
        }

        for layer_name, activations in tqdm(activations_dict.items(), desc="Processing layers"):
            print(f"\nProcessing layer: {layer_name}")

            # Prepare data
            X, y, label_encoder = self.prepare_data(activations, labels, layer_name)

            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_seed
            )

            layer_results = {
                'k_results': {},
                'probe_types': ['dense', 'k_sparse', 'l0_reg'],
                'best_k': None,
                'best_performance': 0
            }

            # Test different sparsity levels
            for k in tqdm(self.config.k_values, desc=f"Testing k values for {layer_name}"):
                k_results = {}

                # Test different probe types
                for probe_type in ['dense', 'k_sparse', 'l0_reg']:
                    if probe_type == 'dense' and k > 1:
                        continue  # Only test dense probe once

                    try:
                        # Train probe
                        probe, train_metrics = self.train_probe(
                            X_train, y_train, k, probe_type
                        )

                        # Evaluate probe
                        eval_metrics = self.evaluate_probe(probe, X_test, y_test)

                        # Store results
                        probe_key = f"{layer_name}_{probe_type}_k{k}"
                        self.trained_probes[probe_key] = probe

                        k_results[probe_type] = {
                            'train_metrics': train_metrics,
                            'eval_metrics': eval_metrics,
                            'probe_key': probe_key
                        }

                        # Update best performance
                        if eval_metrics['accuracy'] > layer_results['best_performance']:
                            layer_results['best_performance'] = eval_metrics['accuracy']
                            layer_results['best_k'] = k

                    except Exception as e:
                        print(f"Error training {probe_type} probe with k={k}: {e}")
                        continue

                layer_results['k_results'][k] = k_results

            experiment_results['layer_results'][layer_name] = layer_results

        # Generate summary
        experiment_results['summary'] = self._generate_experiment_summary(experiment_results)

        return experiment_results

    def _generate_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from experiment results"""
        summary = {
            'best_layers': {},
            'sparsity_analysis': {},
            'probe_type_comparison': defaultdict(list)
        }

        # Find best performing layers and configurations
        for layer_name, layer_data in results['layer_results'].items():
            best_acc = layer_data['best_performance']
            best_k = layer_data['best_k']
            summary['best_layers'][layer_name] = {
                'accuracy': best_acc,
                'best_k': best_k
            }

            # Collect probe type performance
            for k, k_results in layer_data['k_results'].items():
                for probe_type, probe_data in k_results.items():
                    summary['probe_type_comparison'][probe_type].append(
                        probe_data['eval_metrics']['accuracy']
                    )

        # Calculate average performance by probe type
        for probe_type, accuracies in summary['probe_type_comparison'].items():
            summary['probe_type_comparison'][probe_type] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'max_accuracy': np.max(accuracies),
                'min_accuracy': np.min(accuracies)
            }

        return summary

    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create visualizations of the sparse probing results.

        Args:
            results: Experiment results dictionary
            save_path: Optional path to save plots
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Sparse Probing Results: {results['task_name']}", fontsize=16)

        # 1. Accuracy vs Layer for different k values
        ax1 = axes[0, 0]
        layer_names = list(results['layer_results'].keys())

        for k in self.config.k_values:
            accuracies = []
            for layer_name in layer_names:
                layer_data = results['layer_results'][layer_name]
                if k in layer_data['k_results'] and 'k_sparse' in layer_data['k_results'][k]:
                    acc = layer_data['k_results'][k]['k_sparse']['eval_metrics']['accuracy']
                    accuracies.append(acc)
                else:
                    accuracies.append(0)

            ax1.plot(range(len(layer_names)), accuracies, marker='o', label=f'k={k}')

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Layer (K-Sparse Probes)')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([name.split('_')[1] for name in layer_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Sparsity vs Performance
        ax2 = axes[0, 1]
        for layer_name in layer_names[:5]:  # Show first 5 layers to avoid clutter
            layer_data = results['layer_results'][layer_name]
            k_vals = []
            accs = []

            for k in sorted(layer_data['k_results'].keys()):
                if 'k_sparse' in layer_data['k_results'][k]:
                    k_vals.append(k)
                    accs.append(layer_data['k_results'][k]['k_sparse']['eval_metrics']['accuracy'])

            if k_vals:
                ax2.plot(k_vals, accs, marker='s', label=layer_name.split('_')[1])

        ax2.set_xlabel('Sparsity Level (k)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Sparsity Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Probe Type Comparison
        ax3 = axes[1, 0]
        probe_types = list(results['summary']['probe_type_comparison'].keys())
        mean_accs = [results['summary']['probe_type_comparison'][pt]['mean_accuracy']
                    for pt in probe_types]
        std_accs = [results['summary']['probe_type_comparison'][pt]['std_accuracy']
                   for pt in probe_types]

        bars = ax3.bar(probe_types, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)
        ax3.set_ylabel('Mean Accuracy')
        ax3.set_title('Probe Type Comparison')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, mean_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # 4. Best Layer Performance
        ax4 = axes[1, 1]
        best_layers = results['summary']['best_layers']
        layer_names_short = [name.split('_')[1] for name in best_layers.keys()]
        best_accs = [data['accuracy'] for data in best_layers.values()]

        bars = ax4.bar(range(len(layer_names_short)), best_accs, alpha=0.7)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Best Accuracy')
        ax4.set_title('Best Performance by Layer')
        ax4.set_xticks(range(len(layer_names_short)))
        ax4.set_xticklabels(layer_names_short, rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, best_accs)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")

        plt.show()

    import json


    def save_results(self, results, filepath):
        import json
        import torch

        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, '__dict__'):  # for classes like ProbeConfig
                return make_serializable(obj.__dict__)
            elif hasattr(obj, '_asdict'):  # for namedtuples
                return make_serializable(obj._asdict())
            else:
                try:
                    return str(obj)
                except:
                    return f"<<non-serializable: {type(obj).__name__}>>"

        serializable_results = make_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filepath}")



    def _make_serializable(self, obj):
        """Convert torch tensors, numpy arrays, and custom objects to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, ProbeConfig):  # Handle ProbeConfig objects
            return obj.__dict__  # Convert dataclass to dictionary
        else:
            return obj
