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

PROMPT_TEMPLATE_TITLE = (
    "You are a scientific researcher. Based on this human paper title:\n"
    "\"{title}\"\n\n"
    "Write a comprehensive scientific paper abstract (300-500 words) on the same topic that:\n"
    "1) uses appropriate technical terminology,\n"
    "2) maintains academic writing style,\n"
    "3) introduces novel but plausible contributions,\n"
    "4) includes relevant technical details and findings.\n\n"
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
	# Удаляем кодовые блоки и метаданные
	text = re.sub(r"```[\s\S]*?```", " ", text)
	text = re.sub(r"<\/?(code|pre|details|summary)>", " ", text, flags=re.IGNORECASE)
	# Убираем подсказки типа "Final abstract:" и префиксы
	text = re.sub(r"(?i)^\s*(final\s+abstract\s*:|abstract\s*:)", "", text).strip()
	# Ограничим до одной-двух новых строк в начале/конце
	text = text.strip()
	return text


def looks_invalid(text: str, min_chars: int = 300) -> bool:
	if not text:
		return True
	t = text.strip()
	if len(t) < min_chars:
		return True
	# Явные сообщения об ошибке/неподдерживаемой задаче/провайдере
	bad_markers = [
		"Task 'text-generation' not supported",
		"Available tasks:",
		"fireworks-ai",
		"Error:",
		"[Generation error",
		"HTTPException",
		"Traceback (most recent call last)",
	]
	if any(m.lower() in t.lower() for m in bad_markers):
		return True
	# Похоже на код/JSON, а не на абстракт
	code_like_patterns = [r"\bdef\b", r"\bclass\b", r"import ", r"\{\s*\"", r"\}\s*$"]
	if any(re.search(p, t) for p in code_like_patterns):
		return True
	return False


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
		# HF API может вернуть строку, dict или list(dict)
		if isinstance(resp, str):
			text_out = resp
		elif hasattr(resp, 'generated_text'):
			text_out = getattr(resp, 'generated_text')
		elif isinstance(resp, dict):
			text_out = resp.get('generated_text') or resp.get('content') or ""
		elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
			text_out = resp[0].get('generated_text') or resp[0].get('content') or ""
		else:
			text_out = str(resp)
		return sanitize_output(text_out or "")

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
	parser.add_argument("--human_csv", help="Path to CSV with human documents (optional if --human_txt_dir provided)")
	parser.add_argument("--human_txt_dir", help="Path to directory with human TXT files (Title:/Abstract:/Topic:) to drive per-file topics")
	parser.add_argument("--output_dir", required=True, help="Output directory for synthetic docs")
	parser.add_argument("--count", type=int, default=20, help="Number of docs to generate")
	parser.add_argument("--model", required=True, help="HF model id (e.g., meta-llama/Meta-Llama-3.1-70B-Instruct, Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-V3)")
	parser.add_argument("--hf_token", help="Hugging Face token (optional if HF_TOKEN env is set)")
	parser.add_argument("--topic_hint", default="machine learning", help="Topic hint used in prompt if CSV lacks explicit topic")
	parser.add_argument("--sleep_s", type=float, default=0.7, help="Sleep seconds between requests to avoid rate limits")
	parser.add_argument("--mode", choices=["auto", "chat", "text"], default="auto", help="Which API to use for generation")
	parser.add_argument("--stop", nargs="*", default=["</s>", "<|eot_id|>", "<think>", "</think>"], help="Stop sequences for text mode")
	parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample if provider mode unsupported")
	parser.add_argument("--min_chars", type=int, default=300, help="Minimum characters to accept generation as valid")
	args = parser.parse_args()

	# Validate inputs
	if not args.human_csv and not args.human_txt_dir:
		print("Error: provide either --human_csv or --human_txt_dir")
		return
	if args.human_csv and not os.path.exists(args.human_csv):
		print(f"Error: file not found: {args.human_csv}")
		return
	if args.human_txt_dir and not os.path.isdir(args.human_txt_dir):
		print(f"Error: directory not found: {args.human_txt_dir}")
		return
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)

	# Build list of generation items: [{topic, abstract, source}]
	items = []
	if args.human_txt_dir:
		for name in sorted(os.listdir(args.human_txt_dir)):
			if not name.endswith('.txt'):
				continue
			p = os.path.join(args.human_txt_dir, name)
			try:
				with open(p, 'r', encoding='utf-8') as fh:
					content = fh.read()
			except Exception:
				continue
			# Extract Title and Topic (Title is mandatory driver per user spec)
			m_title = re.search(r"(?im)^\s*Title:\s*(.+)$", content)
			title_local = m_title.group(1).strip() if m_title else ""
			m_topic = re.search(r"(?im)^\s*Topic:\s*(.+)$", content)
			topic_local = m_topic.group(1).strip() if m_topic else ""
			if not topic_local:
				# Derive from parent directory name if possible
				dir_base = os.path.basename(args.human_txt_dir).lower()
				if 'text_mining' in dir_base or 'text mining' in dir_base:
					topic_local = 'text mining'
				elif 'information_retrieval' in dir_base or 'information retrieval' in dir_base or 'ir' == dir_base:
					topic_local = 'information retrieval'
				else:
					topic_local = args.topic_hint
			# Extract Abstract:
			abstract = ""
			if 'Abstract:' in content:
				abstract = content.split('Abstract:')[-1].strip()
			else:
				abstract = content.strip()
			if abstract or title_local:
				items.append({'title': title_local, 'topic': topic_local, 'abstract': abstract, 'source': p})
	else:
		# CSV fallback
		try:
			df = pd.read_csv(args.human_csv)
		except Exception as e:
			print(f"Error reading CSV: {e}")
			return
		# Derive topic from csv filename
		topic_from_csv = args.topic_hint
		lc = args.human_csv.lower()
		if "text_mining" in lc:
			topic_from_csv = "text mining"
		elif "information_retrieval" in lc or "information-retrieval" in lc:
			topic_from_csv = "information retrieval"
		for _, row in df.iterrows():
			abstract = str(row.get('abstract', row.get('text', '')))
			if isinstance(abstract, str) and abstract.strip():
				items.append({'topic': topic_from_csv, 'abstract': abstract.strip(), 'source': args.human_csv})

	try:
		client = build_client(args.model, args.hf_token)
	except Exception as e:
		print(f"Error creating HF client: {e}")
		return

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
	# Iterate over items deterministically, generating one synthetic per human file
	for item in items:
		if written >= to_generate:
			break
		attempt_idx += 1
		topic_local = (item.get('topic') or '').strip() or args.topic_hint
		title_local = (item.get('title') or '').strip()
		abstract = (item.get('abstract') or '').strip()
		# Build prompt: prefer Title-only per requirements; fallback to abstract-based
		if title_local:
			prompt = PROMPT_TEMPLATE_TITLE.format(title=title_local)
		else:
			prompt = PROMPT_TEMPLATE.format(topic=topic_local, abstract=abstract)

		# Пробуем выбранный режим, затем альтернативный при ошибке
		text = generate_one(client, prompt, mode=args.mode, stop_sequences=args.stop)
		if is_error_text(text) and args.mode != "chat":
			text = generate_one(client, prompt, mode="chat", stop_sequences=args.stop)
		if is_error_text(text) and args.mode != "text":
			text = generate_one(client, prompt, mode="text", stop_sequences=args.stop)
		text = sanitize_output(text)
		if is_error_text(text) or looks_invalid(text, min_chars=args.min_chars):
			time.sleep(args.sleep_s)
			continue

		written += 1
		fname = f"synthetic_{start_index + written - 1:03d}.txt"
		with open(os.path.join(args.output_dir, fname), "w", encoding="utf-8") as f:
			title_line = title_local if title_local else f"Generated Research Paper on {topic_local.title()}"
			f.write(f"Title: {title_line}\n\nAbstract:\n{text}\n")
		generated_meta.append({
			"file": fname,
			"prompt_chars": int(len(prompt)),
			"output_chars": int(len(text)),
			"source": item.get('source', '')
		})
		time.sleep(args.sleep_s)

	meta = {
		"model": args.model,
		"source_csv": args.human_csv,
		"source_dir": args.human_txt_dir,
		"topic": None,
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
