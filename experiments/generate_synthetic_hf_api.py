#!/usr/bin/env python3
"""
Генерация синтетических документов через Hugging Face Inference API
- Без локальной загрузки весов моделей
- Поддерживает chat_completion и text_generation (настраивается флагом --mode)

Требуется токен доступа HF: переменная окружения HF_TOKEN или аргумент --hf_token
"""

import os
import sys
import time
import json
import argparse
import re
from typing import List
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient

# Добавляем путь к проекту для импорта arxiv_collector при необходимости
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


PROMPT_TEMPLATE = (
	"You are a scientific researcher. Based on this abstract about {topic}:\n\n"
	"\"{abstract}\"\n\n"
	"Write a comprehensive scientific paper abstract (300-500 words) that:\n"
	"1) follows the same research direction and methodology,\n"
	"2) uses similar technical terminology,\n"
	"3) maintains academic writing style,\n"
	"4) introduces novel but plausible research contributions,\n"
	"5) includes relevant technical details and findings.\n\n"
	"Return ONLY the final abstract text (no analysis, no steps, no chain-of-thought).\n"
)


def build_client(model: str, hf_token: str) -> InferenceClient:
	if not hf_token:
		hf_token = os.environ.get("HF_TOKEN")
	if not hf_token:
		raise ValueError("HF token is required. Set HF_TOKEN env var or pass --hf_token")
	return InferenceClient(model=model, token=hf_token)


def sanitize_output(text: str) -> str:
	if not text:
		return text
	# Удаляем возможные блоки размышлений
	text = re.sub(r"```thinking[\s\S]*?```", "", text, flags=re.IGNORECASE)
	text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
	text = re.sub(r"(?i)^(analysis:|reasoning:|thought:|chain\-of\-thought:).*", "", text)
	# Убираем подсказки типа "Final abstract:" и префиксы
	text = re.sub(r"(?i)^\s*(final\s+abstract\s*:|abstract\s*:)", "", text).strip()
	# Ограничим до одной-двух новых строк в начале/конце
	text = text.strip()
	return text


def generate_one(client: InferenceClient, prompt: str, mode: str, max_new_tokens: int = 500, temperature: float = 0.7, top_p: float = 0.95, repetition_penalty: float = 1.05, stop_sequences: List[str] | None = None) -> str:
	def _chat() -> str:
		chat_resp = client.chat_completion(
			messages=[
				{"role": "system", "content": "You are a scientific researcher. Output only the final abstract."},
				{"role": "user", "content": prompt},
			],
			max_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			stream=False,
		)
		choice = chat_resp.choices[0]
		content = choice.get("message", {}).get("content") if isinstance(choice, dict) else getattr(choice.message, "content", None)
		return sanitize_output(content or "")

	def _text() -> str:
		resp = client.text_generation(
			prompt,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			repetition_penalty=repetition_penalty,
			return_full_text=False,
			stream=False,
			stop=stop_sequences or ["</s>", "<|eot_id|>", "<think>", "</think>"]
		)
		return sanitize_output(resp or "")

	if mode == "chat":
		try:
			return _chat()
		except Exception as e:
			return f"[Generation error (chat): {e}]"
	if mode == "text":
		try:
			out = _text()
			# Если провайдер не поддерживает text-generation, пробуем chat
			if "Task 'text-generation' not supported" in out or "fireworks-ai" in out:
				return _chat()
			return out
		except Exception as e:
			msg = str(e)
			if "Task 'text-generation' not supported" in msg or "fireworks-ai" in msg:
				try:
					return _chat()
				except Exception as e2:
					return f"[Generation error (fallback chat): {e2}]"
			return f"[Generation error (text): {e}]"
	# auto
	try:
		return _chat()
	except Exception:
		pass
	try:
		return _text()
	except Exception as e:
		return f"[Generation error: {e}]"


def is_error_text(text: str) -> bool:
	if not text:
		return True
	return text.strip().startswith("[Generation error")


def main():
	parser = argparse.ArgumentParser(description="HF Inference API synthetic generation (no local weights)")
	parser.add_argument("--human_csv", required=True, help="Path to CSV with human documents")
	parser.add_argument("--output_dir", required=True, help="Output directory for synthetic docs")
	parser.add_argument("--count", type=int, default=20, help="Number of docs to generate")
	parser.add_argument("--model", required=True, help="HF model id (e.g., meta-llama/Meta-Llama-3.1-70B-Instruct, Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-V3)")
	parser.add_argument("--hf_token", help="Hugging Face token (optional if HF_TOKEN env is set)")
	parser.add_argument("--topic_hint", default="machine learning", help="Topic hint used in prompt if CSV lacks explicit topic")
	parser.add_argument("--sleep_s", type=float, default=0.7, help="Sleep seconds between requests to avoid rate limits")
	parser.add_argument("--mode", choices=["auto", "chat", "text"], default="auto", help="Which API to use for generation")
	parser.add_argument("--stop", nargs="*", default=["</s>", "<|eot_id|>", "<think>", "</think>"], help="Stop sequences for text mode")
	parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample if provider mode unsupported")
	args = parser.parse_args()

	if not os.path.exists(args.human_csv):
		print(f"Error: file not found: {args.human_csv}")
		return
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)

	try:
		df = pd.read_csv(args.human_csv)
	except Exception as e:
		print(f"Error reading CSV: {e}")
		return

	try:
		client = build_client(args.model, args.hf_token)
	except Exception as e:
		print(f"Error creating HF client: {e}")
		return

	topic = args.topic_hint
	lc = args.human_csv.lower()
	if "text_mining" in lc:
		topic = "text mining"
	elif "information_retrieval" in lc or "information-retrieval" in lc:
		topic = "information retrieval"

	generated_meta = []
	# Поддержка дозаполнения: определяем сколько файлов уже есть
	existing_files = sorted([p for p in os.listdir(args.output_dir) if p.startswith("synthetic_") and p.endswith(".txt")])
	existing_count = len(existing_files)
	start_index = 1
	if existing_count:
		try:
			last = max(int(f.split("_")[-1].split(".")[0]) for f in existing_files)
			start_index = last + 1
		except Exception:
			start_index = existing_count + 1
	to_generate = max(0, args.count - existing_count)
	written = 0
	attempt_idx = 0
	max_rows = len(df) if len(df) > 0 else args.count * 5
	while written < to_generate and attempt_idx < (to_generate or 1) * args.max_retries * 3:
		attempt_idx += 1
		row = df.sample(1).iloc[0] if len(df) > 0 else {"abstract": ""}
		abstract = str(row.get("abstract", row.get("text", "")))
		prompt = PROMPT_TEMPLATE.format(topic=topic, abstract=abstract)

		# Пробуем выбранный режим, затем альтернативный при ошибке
		text = generate_one(client, prompt, mode=args.mode, stop_sequences=args.stop)
		if is_error_text(text) and args.mode != "chat":
			text = generate_one(client, prompt, mode="chat", stop_sequences=args.stop)
		if is_error_text(text) and args.mode != "text":
			text = generate_one(client, prompt, mode="text", stop_sequences=args.stop)
		if is_error_text(text):
			time.sleep(args.sleep_s)
			continue

		written += 1
		fname = f"synthetic_{start_index + written - 1:03d}.txt"
		with open(os.path.join(args.output_dir, fname), "w", encoding="utf-8") as f:
			f.write(f"Title: Generated Research Paper on {topic.title()}\n\nAbstract:\n{text}\n")
		generated_meta.append({
			"file": fname,
			"prompt_chars": int(len(prompt)),
			"output_chars": int(len(text))
		})
		time.sleep(args.sleep_s)

	meta = {
		"model": args.model,
		"source_csv": args.human_csv,
		"topic": topic,
		"count": existing_count + written,
		"requested_count": args.count,
		"api": "huggingface_inference",
		"mode": args.mode,
		"generated": generated_meta,
	}
	with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"Done. Added {written} docs (total {existing_count + written}) to {args.output_dir}")


if __name__ == "__main__":
	main()
