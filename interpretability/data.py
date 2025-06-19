from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict

from dataclasses import dataclass
from tqdm import tqdm


class IOIDatasetProcessor:
    """
    Processor for the Indirect Object Identification (IOI) dataset.
    This version is specifically tailored to the 'fahamu/ioi' dataset format.
    """

    def __init__(self, dataset_name: str = "fahamu/ioi"):
        """
        Initialize the IOI dataset processor.

        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        """Load the IOI dataset from HuggingFace"""
        print(f"Loading dataset: {self.dataset_name}")
        try:
            self.dataset = load_dataset(self.dataset_name)
            print(f"Dataset loaded successfully")
            print(f"Available splits: {list(self.dataset.keys())}")
        except Exception as e:
            print(f"Fatal: Could not load dataset '{self.dataset_name}'. Error: {e}")
            # The program cannot proceed without the correct data.
            raise

    def get_texts_and_labels(self, split: str = "train", n_examples: int = 100) -> Tuple[List[str], List[str]]:
        """
        Get texts and labels for the dataset.
        This method parses the 'ioi_sentences' field to extract the prompt and label.

        Returns:
            Tuple of (texts, labels)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded")

        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(self.dataset.keys())}")

        dataset_split = self.dataset[split]
        num_to_process = min(n_examples, len(dataset_split))
        
        texts = []
        labels = []

        print(f"Processing {num_to_process} examples from the '{split}' split...")
        for i in range(num_to_process):
            full_sentence = dataset_split[i]['ioi_sentences']
            
            # Parsing logic: The last word is the label (indirect object).
            # The rest of the sentence is the prompt.
            words = full_sentence.strip().split(' ')
            if len(words) < 2:
                # Skip malformed sentences if any
                continue

            label = words[-1]
            # The prompt is everything before the last word, with a trailing space.
            text = ' '.join(words[:-1]) + ' '

            texts.append(text)
            labels.append(label)

        return texts, labels