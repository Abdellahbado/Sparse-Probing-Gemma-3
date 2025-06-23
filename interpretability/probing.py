import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = None,
        temperature: float = 2 / 3,
        zeta: float = 1.1,
        gamma: float = -0.1,
    ):
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
            u = torch.rand(
                batch_size,
                self.out_features,
                self.in_features,
                device=self.log_alpha.device,
            )
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature
            )
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
            active_mask = pi > 0.5
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
                    torch.log(
                        torch.tensor(0.99 / 0.01)
                    ),  # High probability for selected
                    torch.log(torch.tensor(0.01 / 0.99)),  # Low probability for others
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

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        k: int = 1,
        probe_type: str = "k_sparse",
        **kwargs,
    ):
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
        if hasattr(self.linear, "get_selected_features"):
            return self.linear.get_selected_features()
        elif hasattr(self.linear, "get_active_neurons"):
            return self.linear.get_active_neurons()
        else:
            # For dense probes, return all features
            return np.arange(self.input_dim)

    def regularization_loss(self):
        """Get regularization loss if applicable"""
        if hasattr(self.linear, "regularization_loss"):
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

    def prepare_data(
        self,
        activations: torch.Tensor,
        labels: List[str],
        attention_mask: torch.Tensor,
        layer_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
        """
        Prepare activation data and labels for probing.

        Args:
            activations: Activation tensor [batch_size, seq_len, hidden_dim]
            labels: List of string labels
            attention_mask: Attention mask tensor [batch_size, seq_len]
            layer_name: Name of the layer being probed

        Returns:
            Tuple of (processed_activations, encoded_labels, label_encoder)
        """
        # Convert activations to float32 to avoid dtype mismatches
        activations = activations.to(dtype=torch.float32)

        # Handle different activation shapes
        if len(activations.shape) == 3:
            # Use the attention mask to find the index of the last non-padding token.
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = (sequence_lengths - 1).long()

            # Gather the activations from the last token position for each item in the batch.
            batch_indices = torch.arange(activations.size(0), device=activations.device)
            activations = activations[batch_indices, last_token_indices, :]

        # Encode labels
        if layer_name not in self.label_encoders:
            self.label_encoders[layer_name] = LabelEncoder()
            encoded_labels = self.label_encoders[layer_name].fit_transform(labels)
        else:
            encoded_labels = self.label_encoders[layer_name].transform(labels)

        return (
            activations,
            torch.tensor(encoded_labels, dtype=torch.long),
            self.label_encoders[layer_name],
        )

    def train_probe(
        self, X: torch.Tensor, y: torch.Tensor, k: int, probe_type: str = "k_sparse"
    ) -> Tuple[SparseProbe, Dict[str, float]]:
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
            X.cpu().numpy(),
            y.cpu().numpy(),
            test_size=0.2,
            random_state=self.config.random_seed,
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
                batch_X = X_train[i : i + self.config.batch_size]
                batch_y = y_train[i : i + self.config.batch_size]

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
        if hasattr(probe.linear, "enforce_k_sparsity"):
            probe.linear.enforce_k_sparsity()

        training_metrics = {
            "best_val_acc": best_val_acc,
            "final_train_loss": train_losses[-1],
            "epochs_trained": epoch + 1,
            "train_losses": train_losses,
            "val_accs": val_accs,
        }

        return probe, training_metrics

    def evaluate_probe(
        self, probe: SparseProbe, X_test: torch.Tensor, y_test: torch.Tensor
    ) -> Dict[str, Any]:
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
                y_true, y_pred, average="weighted", zero_division=0
            )

            # Get active features
            active_features = probe.get_active_features()

            # Correctly calculate num_active_features for all probe types
            num_active = 0
            if probe.probe_type == "l0_reg":
                # For L0, active_features is a tuple of tensors (rows, cols).
                # We count the number of unique input features (cols) that are active.
                if isinstance(active_features, tuple) and len(active_features) > 1:
                    unique_input_features = torch.unique(active_features[1])
                    num_active = len(unique_input_features)
            elif isinstance(active_features, np.ndarray):
                # For k-sparse and dense probes
                num_active = len(active_features)

            sparsity_ratio = num_active / probe.input_dim if probe.input_dim > 0 else 0

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "active_features": active_features,
                "num_active_features": num_active,
                "sparsity_ratio": sparsity_ratio,
            }

        return metrics

    def run_sparse_probing_experiment(
        self,
        activations_dict: Dict[str, torch.Tensor],
        labels: List[str],
        attention_mask: torch.Tensor,
        task_name: str = "classification",
    ) -> Dict[str, Any]:
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
            "task_name": task_name,
            "config": self.config,
            "layer_results": {},
            "summary": {},
        }

        for layer_name, activations in tqdm(
            activations_dict.items(), desc="Processing layers"
        ):
            print(f"\nProcessing layer: {layer_name}")

            # Prepare data
            X, y, label_encoder = self.prepare_data(
                activations, labels, attention_mask, layer_name
            )

            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_seed
            )

            layer_results = {
                "k_results": {},
                "probe_types": ["dense", "k_sparse", "l0_reg"],
                "best_k": None,
                "best_performance": 0,
            }

            # Test different sparsity levels
            for k in tqdm(
                self.config.k_values, desc=f"Testing k values for {layer_name}"
            ):
                k_results = {}

                # Test different probe types
                for probe_type in ["dense", "k_sparse", "l0_reg"]:
                    if probe_type == "dense" and k > 1:
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
                            "train_metrics": train_metrics,
                            "eval_metrics": eval_metrics,
                            "probe_key": probe_key,
                        }

                        # Update best performance
                        if eval_metrics["accuracy"] > layer_results["best_performance"]:
                            layer_results["best_performance"] = eval_metrics["accuracy"]
                            layer_results["best_k"] = k

                    except Exception as e:
                        print(f"Error training {probe_type} probe with k={k}: {e}")
                        continue

                layer_results["k_results"][k] = k_results

            experiment_results["layer_results"][layer_name] = layer_results

        # Generate summary
        experiment_results["summary"] = self._generate_experiment_summary(
            experiment_results
        )

        return experiment_results

    def _generate_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from experiment results"""
        summary = {
            "best_layers": {},
            "probe_type_comparison": defaultdict(list),
            "k_values": self.config.k_values,
        }

        # Add num_classes to the summary from the first available label encoder
        if self.label_encoders:
            first_encoder_key = next(iter(self.label_encoders))
            summary["num_classes"] = len(
                self.label_encoders[first_encoder_key].classes_
            )
        else:
            summary["num_classes"] = 0  # Fallback

        # Find best performing layers and configurations
        for layer_name, layer_data in results["layer_results"].items():
            best_acc = -1.0
            best_k = -1
            best_probe = "N/A"

            # Collect probe type performance and find the best config for the layer
            for k, k_results in layer_data["k_results"].items():
                for probe_type, probe_data in k_results.items():
                    acc = probe_data["eval_metrics"]["accuracy"]
                    summary["probe_type_comparison"][probe_type].append(acc)

                    if acc > best_acc:
                        best_acc = acc
                        best_k = k
                        best_probe = probe_type

            summary["best_layers"][layer_name] = {
                "accuracy": best_acc,
                "best_k": best_k,
                "best_probe": best_probe,
            }

        # Calculate average performance by probe type
        for probe_type, accuracies in summary["probe_type_comparison"].items():
            if accuracies:
                summary["probe_type_comparison"][probe_type] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "max_accuracy": np.max(accuracies),
                    "min_accuracy": np.min(accuracies),
                }
            else:
                summary["probe_type_comparison"][probe_type] = {
                    "mean_accuracy": 0,
                    "std_accuracy": 0,
                    "max_accuracy": 0,
                    "min_accuracy": 0,
                }

        return summary

    def visualize_results(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """
        Create and save visualizations of the probing results.

        Args:
            results: Experiment results dictionary
            save_path: Optional path to save plots
        """
        layer_names = list(results["layer_results"].keys())
        if not layer_names:
            print("No layer results to visualize.")
            return

        # Determine random chance accuracy
        num_classes = results["summary"]["num_classes"]
        random_chance = 1 / num_classes if num_classes > 0 else 0

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f"Sparse Probing Results: {results['task_name']}", fontsize=16)

        # 1. Accuracy vs Layer (Comparing Probe Types) - IMPROVED
        ax1 = axes[0, 0]
        probe_types = ["dense", "k_sparse", "l0_reg"]
        # Use the largest k for sparse probes as a representative example
        k_values = results["summary"]["k_values"]
        k_rep = max(k_values)
        k_first = min(k_values)

        for probe_type in probe_types:
            accuracies = []
            for layer_name in layer_names:
                k_res = results["layer_results"][layer_name]["k_results"]
                if probe_type == "dense":
                    # Dense probe accuracy is the same for all k, stored under the first k
                    if k_first in k_res and "dense" in k_res[k_first]:
                        acc = k_res[k_first]["dense"]["eval_metrics"]["accuracy"]
                    else:
                        acc = np.nan
                else:
                    # Use representative k for sparse probes
                    if k_rep in k_res and probe_type in k_res[k_rep]:
                        acc = k_res[k_rep][probe_type]["eval_metrics"]["accuracy"]
                    else:
                        acc = np.nan
                accuracies.append(acc)
            label = "dense" if probe_type == "dense" else f"{probe_type} (k={k_rep})"
            ax1.plot(
                layer_names,
                accuracies,
                marker="o",
                linestyle="-",
                label=label,
            )

        ax1.axhline(
            y=random_chance,
            color="r",
            linestyle="--",
            label=f"Random Chance ({random_chance:.2f})",
        )
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy vs. Layer by Probe Type")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy vs Sparsity Level (k)
        ax2 = axes[0, 1]
        for layer_name in layer_names:
            layer_data = results["layer_results"][layer_name]
            k_vals = sorted(layer_data["k_results"].keys())
            accs = [
                layer_data["k_results"][k]["k_sparse"]["eval_metrics"]["accuracy"]
                for k in k_vals
                if "k_sparse" in layer_data["k_results"][k]
            ]
            ax2.plot(
                k_vals,
                accs,
                marker="s",
                linestyle="--",
                label=f"Layer {layer_name.split('_')[1]}",
            )
        ax2.set_xlabel("Sparsity Level (k)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Sparsity Level (k-sparse)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Probe Type Comparison (Mean Accuracy)
        ax3 = axes[1, 0]
        probe_comparison = results["summary"]["probe_type_comparison"]
        types = list(probe_comparison.keys())
        mean_accs = [probe_comparison[t]["mean_accuracy"] for t in types]
        std_accs = [probe_comparison[t]["std_accuracy"] for t in types]
        ax3.bar(types, mean_accs, yerr=std_accs, capsize=5, alpha=0.7)
        ax3.set_ylabel("Mean Accuracy")
        ax3.set_title("Probe Type Comparison (Averaged)")
        ax3.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(mean_accs):
            ax3.text(i, v + 0.01, f"{v:.3f}", ha="center")

        # 4. Best Performance by Layer - IMPROVED
        ax4 = axes[1, 1]
        best_layers = results["summary"]["best_layers"]
        layers = list(best_layers.keys())
        best_accs = [d["accuracy"] for d in best_layers.values()]

        bars = ax4.bar(layers, best_accs, alpha=0.7)
        ax4.set_xlabel("Layer")
        ax4.set_ylabel("Best Accuracy")
        ax4.set_title("Best Performance by Layer")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, axis="y", alpha=0.3)

        for i, bar in enumerate(bars):
            data = list(best_layers.values())[i]
            yval = bar.get_height()
            # Add annotation for which probe type and k achieved the best score
            if data['best_probe'] == 'dense':
                label = f"{yval:.3f}\n(dense)"
            else:
                label = f"{yval:.3f}\n({data['best_probe']} @ k={data['best_k']})"
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                label,
                va="bottom",
                ha="center",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plots saved to {save_path}")

        plt.show()

    def visualize_advanced_analysis(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """
        Create advanced visualizations focusing on feature overlap and L0 probe behavior.
        IMPROVED VERSION for clarity and readability.
        """
        layer_names = list(results["layer_results"].keys())
        if not layer_names:
            print("No layer results to visualize.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(
            f"Advanced Probing Analysis: {results['task_name']}", fontsize=18, y=1.02
        )

        # Helper function for Jaccard Similarity
        def jaccard_similarity(set1, set2):
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

        # --- Plot 1: Feature Overlap (IMPROVED with Bar Chart) ---
        ax1 = axes[0]
        k_vals = sorted(results["summary"]["k_values"])
        mean_similarities = []
        std_similarities = []

        for k in k_vals:
            similarities_for_k = []
            for layer_name in layer_names:
                k_res = results["layer_results"][layer_name]["k_results"].get(k, {})
                if "k_sparse" in k_res and "l0_reg" in k_res:
                    k_sparse_features = set(
                        k_res["k_sparse"]["eval_metrics"]["active_features"]
                    )
                    l0_reg_raw = k_res["l0_reg"]["eval_metrics"]["active_features"]
                    l0_reg_features = (
                        set(l0_reg_raw[1].tolist())
                        if isinstance(l0_reg_raw, tuple)
                        else set()
                    )
                    similarities_for_k.append(
                        jaccard_similarity(k_sparse_features, l0_reg_features)
                    )

            if similarities_for_k:
                mean_similarities.append(np.mean(similarities_for_k))
                std_similarities.append(np.std(similarities_for_k))
            else:
                mean_similarities.append(0)
                std_similarities.append(0)

        ax1.bar(
            range(len(k_vals)),
            mean_similarities,
            yerr=std_similarities,
            capsize=5,
            color="skyblue",
            alpha=0.8,
        )
        ax1.set_xticks(range(len(k_vals)))
        ax1.set_xticklabels(k_vals)
        ax1.set_xlabel("Target Sparsity (k)")
        ax1.set_ylabel("Mean Jaccard Similarity")
        ax1.set_title("Feature Overlap: k-sparse vs l0-reg (Avg)")
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, axis="y", alpha=0.3)

        # --- Plot 2: L0 Probe Performance (IMPROVED with connected lines) ---
        ax2 = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
        for i, layer_name in enumerate(layer_names):
            layer_data = results["layer_results"][layer_name]
            sorted_k = sorted(layer_data["k_results"].keys())

            learned_sparsities = [
                layer_data["k_results"][k]["l0_reg"]["eval_metrics"][
                    "num_active_features"
                ]
                for k in sorted_k
                if "l0_reg" in layer_data["k_results"][k]
            ]
            accuracies = [
                layer_data["k_results"][k]["l0_reg"]["eval_metrics"]["accuracy"]
                for k in sorted_k
                if "l0_reg" in layer_data["k_results"][k]
            ]

            if learned_sparsities:
                ax2.plot(
                    learned_sparsities,
                    accuracies,
                    marker="o",
                    linestyle="--",
                    color=colors[i],
                    label=f"L{layer_name.split('_')[1]}",
                    alpha=0.6,
                )

        ax2.set_xlabel("Number of Active Features (Learned by L0)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("L0 Probe: Accuracy vs. Learned Sparsity")
        ax2.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Probe Efficiency (IMPROVED by showing only best layer) ---
        ax3 = axes[2]
        best_layer_info = max(
            results["summary"]["best_layers"].items(),
            key=lambda item: item[1]["accuracy"],
        )
        best_layer_name = best_layer_info[0]

        layer_data = results["layer_results"][best_layer_name]
        k_vals = sorted(layer_data["k_results"].keys())

        # K-Sparse data for best layer
        k_sparse_x = [k for k in k_vals if "k_sparse" in layer_data["k_results"][k]]
        k_sparse_y = [
            layer_data["k_results"][k]["k_sparse"]["eval_metrics"]["accuracy"]
            for k in k_sparse_x
        ]

        # L0-Reg data for best layer
        l0_x = [
            res["l0_reg"]["eval_metrics"]["num_active_features"]
            for k, res in layer_data["k_results"].items()
            if "l0_reg" in res
        ]
        l0_y = [
            res["l0_reg"]["eval_metrics"]["accuracy"]
            for k, res in layer_data["k_results"].items()
            if "l0_reg" in res
        ]

        # Dense probe performance for best layer
        dense_acc = np.nan  # Default to NaN
        if (
            k_vals
            and k_vals[0] in layer_data["k_results"]
            and "dense" in layer_data["k_results"][k_vals[0]]
        ):
            dense_acc = layer_data["k_results"][k_vals[0]]["dense"]["eval_metrics"][
                "accuracy"
            ]

        ax3.plot(
            k_sparse_x,
            k_sparse_y,
            marker="o",
            linestyle="--",
            color="darkcyan",
            label="k-sparse",
        )
        ax3.scatter(l0_x, l0_y, marker="x", color="crimson", s=100, label="l0-reg")
        if not np.isnan(dense_acc):
            ax3.axhline(
                y=dense_acc,
                color="black",
                linestyle=":",
                label=f"Dense Probe ({dense_acc:.2f})",
            )

        ax3.set_xlabel("Number of Active Features")
        ax3.set_ylabel("Accuracy")
        ax3.set_title(f"Probe Efficiency on Best Layer ({best_layer_name})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for legend

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Advanced analysis plots saved to {save_path}")

        plt.show()

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
            elif hasattr(obj, "__dict__"):  # for classes like ProbeConfig
                return make_serializable(obj.__dict__)
            elif hasattr(obj, "_asdict"):  # for namedtuples
                return make_serializable(obj._asdict())
            else:
                try:
                    return str(obj)
                except:
                    return f"<<non-serializable: {type(obj).__name__}>>"

        serializable_results = make_serializable(results)

        with open(filepath, "w") as f:
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
