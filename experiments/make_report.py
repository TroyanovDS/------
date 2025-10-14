import os
import json
import argparse
import pandas as pd


def load_metrics(path: str):
	csv = os.path.join(path, 'metrics.csv')
	json_path = os.path.join(path, 'metrics.json')
	df = pd.read_csv(csv) if os.path.exists(csv) else None
	with open(json_path, 'r', encoding='utf-8') as f:
		js = json.load(f)
	return df, js


def make_markdown(exp1_dir: str, exp2_dir: str, out_path: str):
	df1, js1 = load_metrics(exp1_dir)
	df2, js2 = load_metrics(exp2_dir)
	lines = []
	lines.append('# Experiment Report\n')
	lines.append('## Experiment 1: TF-IDF (n-grams) + MLP\n')
	lines.append(df1.to_markdown(index=False) + '\n')
	plot1 = os.path.join(exp1_dir, 'plots', 'tfidf_mlp_metrics.png')
	if os.path.exists(plot1):
		lines.append(f'![TF-IDF metrics]({plot1})\n')
	lines.append('\n')
	lines.append('## Experiment 2: BERT-family embeddings + MLP\n')
	lines.append(df2.to_markdown(index=False) + '\n')
	plot2 = os.path.join(exp2_dir, 'plots', 'bert_mlp_metrics.png')
	if os.path.exists(plot2):
		lines.append(f'![BERT metrics]({plot2})\n')
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(lines))
	print(f"Saved report to {out_path}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp1', required=True)
	parser.add_argument('--exp2', required=True)
	parser.add_argument('--out', required=True)
	args = parser.parse_args()
	make_markdown(args.exp1, args.exp2, args.out)
