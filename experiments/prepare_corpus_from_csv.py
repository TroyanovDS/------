import os
import argparse
import pandas as pd
from pathlib import Path


def write_txts(df: pd.DataFrame, out_dir: str, prefix: str):
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	for i, row in df.iterrows():
		title = str(row.get('title', ''))
		abstract = str(row.get('abstract', ''))
		text = f"Title: {title}\n\nAbstract: {abstract}\n"
		fname = os.path.join(out_dir, f"{prefix}_{i+1:03d}.txt")
		with open(fname, 'w', encoding='utf-8') as f:
			f.write(text)


def main(csv_text_mining: str, csv_ir: str, out_root: str = 'data/human'):
	os.makedirs(out_root, exist_ok=True)
	df_tm = pd.read_csv(csv_text_mining)
	df_ir = pd.read_csv(csv_ir)
	write_txts(df_tm, os.path.join(out_root, 'text_mining'), 'tm')
	write_txts(df_ir, os.path.join(out_root, 'information_retrieval'), 'ir')
	print(f"Wrote TXT corpus into {out_root}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_text_mining', default='data/text_mining.csv')
	parser.add_argument('--csv_ir', default='data/information_retrieval.csv')
	parser.add_argument('--out_root', default='data/human')
	args = parser.parse_args()
	main(args.csv_text_mining, args.csv_ir, args.out_root)
