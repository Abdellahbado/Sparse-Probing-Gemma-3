import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import gc
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
from .config import HF_TOKEN, MODEL_NAME
 

@dataclass
class ActivationHook:
    """Stores information about registered hooks"""
    name: str
    layer_idx: int
    component: str
    hook_fn: Callable
    handle: Any = None

class MechanisticInterpretabilityFramework:
    """
    Framework for mechanistic interpretability research with Gemma models.
    Provides infrastructure for activation extraction, hooks, and batch processing.
    """

    def __init__(self, model_name: str = "google/gemma-1.1-2b", device: str = "auto"):
        """
        Initialize the framework with a Gemma model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)

        # Load model and tokenizer
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device != "auto" else None,
            trust_remote_code=True,
            token=HF_TOKEN
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Storage for activations and hooks
        self.activations = defaultdict(list)
        self.hooks = []

        # Model architecture info
        self.n_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size

        print(f"Model loaded: {self.n_layers} layers, hidden size {self.hidden_size}")

    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def create_activation_hook(self, layer_idx: int, component: str = "output") -> Callable:
        """
        Create a hook function to capture activations.

        Args:
            layer_idx: Which layer to hook
            component: Which component to hook ('output', 'mlp', 'attention')

        Returns:
            Hook function
        """
        hook_name = f"layer_{layer_idx}_{component}"

        def hook_fn(module, input, output):
            # Store the activation
            if isinstance(output, tuple):
                activation = output[0]  # Usually the first element is the main output
            else:
                activation = output

            # Convert to CPU to save memory if on GPU
            if activation.device.type == 'cuda':
                activation = activation.cpu()

            self.activations[hook_name].append(activation.detach())

        return hook_fn

    def register_hooks(self, layer_indices: List[int], components: List[str] = ["output"]):
        """
        Register hooks for specified layers and components.

        Args:
            layer_indices: List of layer indices to hook
            components: List of components to hook per layer
        """
        self.clear_hooks()  # Clear existing hooks

        for layer_idx in layer_indices:
            if layer_idx >= self.n_layers:
                print(f"Warning: Layer {layer_idx} doesn't exist (max: {self.n_layers-1})")
                continue

            layer = self.model.model.layers[layer_idx]

            for component in components:
                hook_fn = self.create_activation_hook(layer_idx, component)

                if component == "output":
                    handle = layer.register_forward_hook(hook_fn)
                elif component == "mlp":
                    handle = layer.mlp.register_forward_hook(hook_fn)
                elif component == "attention":
                    handle = layer.self_attn.register_forward_hook(hook_fn)
                else:
                    print(f"Warning: Unknown component '{component}'")
                    continue

                hook_info = ActivationHook(
                    name=f"layer_{layer_idx}_{component}",
                    layer_idx=layer_idx,
                    component=component,
                    hook_fn=hook_fn,
                    handle=handle
                )
                self.hooks.append(hook_info)

        print(f"Registered {len(self.hooks)} hooks")

    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            if hook.handle is not None:
                hook.handle.remove()
        self.hooks.clear()
        print("Cleared all hooks")

    def clear_activations(self):
        """Clear stored activations to free memory"""
        self.activations.clear()
        gc.collect()

    def forward_with_cache(self, input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None,
                          max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect activations.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            max_length: Maximum sequence length to pad/truncate activations to

        Returns:
            Dictionary mapping hook names to activations
        """
        self.clear_activations()

        with torch.no_grad():
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Move inputs to model device
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # Forward pass (activations are captured by hooks)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Convert list of tensors to single tensor for each hook, ensuring consistent sequence length
        cached_activations = {}
        for hook_name, activations in self.activations.items():
            if activations:
                # Stack activations and pad/truncate to max_length
                stacked_activations = []
                for activation in activations:
                    # Ensure activation is [batch_size, seq_len, hidden_size]
                    if len(activation.shape) == 2:
                        activation = activation.unsqueeze(1)  # Add seq_len dimension if needed
                    seq_len = activation.shape[1]
                    if seq_len > max_length:
                        # Truncate to max_length
                        activation = activation[:, :max_length, :]
                    elif seq_len < max_length:
                        # Pad to max_length with zeros
                        pad_size = max_length - seq_len
                        activation = torch.nn.functional.pad(
                            activation, (0, 0, 0, pad_size), mode='constant', value=0
                        )
                    stacked_activations.append(activation)
                # Concatenate along batch dimension
                cached_activations[hook_name] = torch.cat(stacked_activations, dim=0)

        return cached_activations

    def extract_activations(self, texts: List[str], batch_size: int = 8,
                           max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping hook names to activation tensors
        """
        all_activations = defaultdict(list)

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Get activations for this batch
            batch_activations = self.forward_with_cache(
                encoded["input_ids"],
                encoded["attention_mask"]
            )

            # Accumulate activations
            for hook_name, activations in batch_activations.items():
                all_activations[hook_name].append(activations)

        # Concatenate all batches
        final_activations = {}
        for hook_name, activation_list in all_activations.items():
            final_activations[hook_name] = torch.cat(activation_list, dim=0)

        return final_activations
