#!/usr/bin/env python3
"""
Синтетическая генерация через провайдерские API (без HF Inference Router)

Поддерживаемые провайдеры и ключи окружения:
- deepseek:   DEEPSEEK_API_KEY (https://api.deepseek.com/chat/completions)
- dashscope:  DASHSCOPE_API_KEY (https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions)
- openrouter: OPENROUTER_API_KEY (https://openrouter.ai/api/v1/chat/completions)

Пример путей вывода (совместимы с текущим анализом):
- Llama (openrouter): data/ai/llama_api_text/text_mining_full, data/ai/llama_api/ir
- Qwen  (dashscope):  data/ai/qwen_api_auto/text_mining_full, data/ai/qwen_api/ir
- DeepSeek-R1 (deepseek): data/ai/deepseek_api_auto/text_mining_full, data/ai/deepseek_api/ir
"""

import os
import sys
import re
import time
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import requests


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


def sanitize_output(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"```thinking[\s\S]*?```", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*?</think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)^(analysis:|reasoning:|thought:|chain\-of\-thought:).*", " ", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"<\/?(code|pre|details|summary)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)^\s*(final\s+abstract\s*:|abstract\s*:)", "", text).strip()
    return text.strip()


def looks_invalid(text: str, min_chars: int = 300) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < min_chars:
        return True
    bad_markers = [
        "Task 'text-generation' not supported",
        "Available tasks:",
        "payment required",
        "[Generation error",
        "HTTPException",
        "Traceback (most recent call last)",
    ]
    if any(m.lower() in t.lower() for m in bad_markers):
        return True
    code_like_patterns = [r"\bdef\b", r"\bclass\b", r"import ", r"\{\s*\"", r"\}\s*$"]
    if any(re.search(p, t) for p in code_like_patterns):
        return True
    return False


def post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def parse_chat_response(obj: Dict[str, Any]) -> str:
    # Унифицированный парсер OpenAI-совместимых ответов
    try:
        choices = obj.get("choices") or []
        if not choices:
            return ""
        ch0 = choices[0]
        msg = ch0.get("message") or {}
        content = msg.get("content")
        if content:
            return str(content)
        # Иногда текст в поле 'text'/'generated_text'
        return str(ch0.get("text") or ch0.get("generated_text") or "")
    except Exception:
        return ""


def call_deepseek(prompt: str, api_key: str, model: str, max_tokens: int = 600) -> str:
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a scientific researcher. Output only the final abstract."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    data = post_json(url, headers, payload)
    return parse_chat_response(data)


def call_dashscope(prompt: str, api_key: str, model: str, max_tokens: int = 600) -> str:
    # OpenAI-совместимый чат-эндпоинт DashScope
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "qwen2.5-72b-instruct",
        "messages": [
            {"role": "system", "content": "You are a scientific researcher. Output only the final abstract."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    data = post_json(url, headers, payload)
    return parse_chat_response(data)


def call_openrouter(prompt: str, api_key: str, model: str, max_tokens: int = 600) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "meta-llama/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a scientific researcher. Output only the final abstract."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    data = post_json(url, headers, payload)
    return parse_chat_response(data)


def main():
    parser = argparse.ArgumentParser(description="Synthetic generation via provider APIs")
    parser.add_argument("--human_csv")
    parser.add_argument("--human_txt_dir", help="Path to directory with human TXT files (Title:/Abstract:/Topic:) to drive per-file topics")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", choices=["deepseek", "dashscope", "openrouter"], required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--topic_hint", default="machine learning")
    parser.add_argument("--sleep_s", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--min_chars", type=int, default=300)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build items list: prefer per-file TXT with Title, fallback to CSV abstracts
    items = []
    if args.human_txt_dir:
        if not os.path.isdir(args.human_txt_dir):
            print(f"Error: directory not found: {args.human_txt_dir}")
            return
        for name in sorted(os.listdir(args.human_txt_dir)):
            if not name.endswith('.txt'):
                continue
            p = os.path.join(args.human_txt_dir, name)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    content = fh.read()
            except Exception:
                continue
            m_title = re.search(r"(?im)^\s*Title:\s*(.+)$", content)
            title_local = m_title.group(1).strip() if m_title else ""
            m_topic = re.search(r"(?im)^\s*Topic:\s*(.+)$", content)
            topic_local = m_topic.group(1).strip() if m_topic else ""
            abstract = ""
            if 'Abstract:' in content:
                abstract = content.split('Abstract:')[-1].strip()
            else:
                abstract = content.strip()
            if title_local or abstract:
                items.append({'title': title_local, 'topic': topic_local, 'abstract': abstract, 'source': p})
    else:
        try:
            df = pd.read_csv(args.human_csv)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        topic_from_csv = args.topic_hint
        lc = (args.human_csv or '').lower()
        if "text_mining" in lc:
            topic_from_csv = "text mining"
        elif "information_retrieval" in lc or "information-retrieval" in lc:
            topic_from_csv = "information retrieval"
        for _, row in df.iterrows():
            abstract = str(row.get('abstract', row.get('text', '')))
            if isinstance(abstract, str) and abstract.strip():
                items.append({'title': '', 'topic': topic_from_csv, 'abstract': abstract.strip(), 'source': args.human_csv})

    # API keys
    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        caller = lambda p: call_deepseek(p, api_key, args.model)
    elif args.provider == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        caller = lambda p: call_dashscope(p, api_key, args.model)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        caller = lambda p: call_openrouter(p, api_key, args.model)

    if not api_key:
        print(f"Error: missing API key for provider {args.provider}. Set the corresponding environment variable.")
        return

    # Detect how many already exist
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
    attempts = 0
    max_attempts = max(1, to_generate) * args.max_retries * 4

    while written < to_generate and attempts < max_attempts:
        attempts += 1
        item = items[(written + attempts - 1) % len(items)] if items else {"title":"","topic":args.topic_hint,"abstract":""}
        title_local = (item.get('title') or '').strip()
        topic_local = (item.get('topic') or '').strip() or args.topic_hint
        abstract = (item.get('abstract') or '').strip()
        if title_local:
            prompt = PROMPT_TEMPLATE_TITLE.format(title=title_local)
        else:
            prompt = PROMPT_TEMPLATE.format(topic=topic_local, abstract=abstract)

        try:
            text = caller(prompt)
        except Exception as e:
            print(f"Call error: {e}")
            time.sleep(args.sleep_s)
            continue

        text = sanitize_output(text)
        if looks_invalid(text, min_chars=args.min_chars):
            time.sleep(args.sleep_s)
            continue

        written += 1
        fname = f"synthetic_{start_index + written - 1:03d}.txt"
        with open(os.path.join(args.output_dir, fname), "w", encoding="utf-8") as f:
            title_line = title_local if title_local else f"Generated Research Paper on {topic_local.title()}"
            f.write(f"Title: {title_line}\n\nAbstract:\n{text}\n")
        time.sleep(args.sleep_s)

    meta = {
        "provider": args.provider,
        "model": args.model,
        "source_csv": args.human_csv,
        "topic": topic,
        "count": existing_count + written,
        "requested_count": args.count,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Done. Added {written} docs (total {existing_count + written}) to {args.output_dir}")


if __name__ == "__main__":
    main()


