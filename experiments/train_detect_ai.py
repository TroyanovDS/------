import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt


def load_features(dir_path: str) -> pd.DataFrame:
	csv = os.path.join(dir_path, 'features.csv')
	return pd.read_csv(csv, index_col=0)


def train_and_eval(X: np.ndarray, y: np.ndarray, model_type: str = 'logreg') -> dict:
	Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
	if model_type == 'mlp':
		clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, random_state=42)
	else:
		clf = LogisticRegression(max_iter=1000, n_jobs=1)
	clf.fit(Xtr, ytr)
	pred = clf.predict(Xte)
	proba = clf.predict_proba(Xte)[:,1] if hasattr(clf, 'predict_proba') else pred
	acc = accuracy_score(yte, pred)
	prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average='binary')
	try:
		auc = roc_auc_score(yte, proba)
	except Exception:
		auc = float('nan')
	return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auc": float(auc)}


def run_scenario(human_dir: str, ai_dir: str, out_dir: str, model_type: str = 'logreg'):
	os.makedirs(out_dir, exist_ok=True)
	human = load_features(human_dir)
	ai = load_features(ai_dir)
	# Align columns
	cols = sorted(set(human.columns) & set(ai.columns))
	human = human[cols]
	ai = ai[cols]
	X = np.vstack([human.values, ai.values])
	y = np.array([0]*len(human) + [1]*len(ai), dtype=np.int64)
	metrics = train_and_eval(X, y, model_type)
	with open(os.path.join(out_dir, f'metrics_{model_type}.json'), 'w', encoding='utf-8') as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)
	print(f"Saved metrics to {out_dir}")


def main():
	parser = argparse.ArgumentParser(description='Train AI-text detector on keyword features')
	parser.add_argument('--human_features', required=True)
	parser.add_argument('--ai_features', nargs='+', required=True, help='One or more AI feature dirs (deepseek/llama/qwen)')
	parser.add_argument('--out_root', required=True)
	parser.add_argument('--model', choices=['logreg','mlp'], default='logreg')
	args = parser.parse_args()

	for ai_dir in args.ai_features:
		name = os.path.basename(os.path.normpath(ai_dir))
		out_dir = os.path.join(args.out_root, f'{name}_{args.model}')
		run_scenario(args.human_features, ai_dir, out_dir, args.model)


if __name__ == '__main__':
	main()


