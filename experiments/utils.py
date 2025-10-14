import os
import json
from typing import List, Tuple
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_texts_from_root(root: str, limit: int = 20) -> List[str]:
	texts: List[str] = []
	count = 0
	for dirpath, _, files in os.walk(root):
		for f in files:
			if f.lower().endswith('.txt'):
				with open(os.path.join(dirpath, f), 'r', encoding='utf-8') as fh:
					texts.append(fh.read())
					count += 1
					if count >= limit:
						return texts
	return texts


def build_dataset(human_root: str, synthetic_root: str, limit: int = 20) -> Tuple[List[str], np.ndarray]:
	human = load_texts_from_root(human_root, limit)
	synth = load_texts_from_root(synthetic_root, limit)
	X = human + synth
	y = np.array([0] * len(human) + [1] * len(synth), dtype=np.int64)
	return X, y


def split_dataset(X: List[str], y: np.ndarray, test_size: float = 0.3, random_state: int = 42):
	return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Stub for open LLM generation via HF if needed later
class OpenLLMGenerator:
	def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
		self.model_name = model_name

	def generate(self, prompts: List[str]) -> List[str]:
		# Placeholder: integrate HF pipelines or TGI client if available
		return [f"[LLM {self.model_name}] Synthetic text for: {p[:60]}..." for p in prompts]
