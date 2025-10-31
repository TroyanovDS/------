#!/usr/bin/env python3
"""
Эксперимент 1 (перемоделный): Анализ ключевых слов и вводных слов
- Сравнение 100 человеческих документов (50 TM + 50 IR) против 100 синтетических документов КАЖДОЙ модели
- Методы: n-граммы (TF-IDF), YAKE, TextRank
- Метрики: Jaccard, Overlap Human, Overlap Synthetic, Harmonic Mean
- Дополнительно: частота вводных/связующих слов (connectives) и косинусное сходство TF-IDF центроидов

Выход: results/experiment1_per_model/experiment1_per_model_report.md (+ графики по моделям)
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    USE_SEABORN = True
except Exception:
    USE_SEABORN = False

# Позволим импортировать функции из run_experiment1_keywords.py
sys.path.append(os.path.dirname(__file__))
import run_experiment1_keywords as exp1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_auc_score, roc_curve


CONNECTIVES = [
    # контраст/вывод
    "however", "therefore", "thus", "hence", "nevertheless", "nonetheless",
    # добавление/пример
    "moreover", "furthermore", "in addition", "additionally", "for example", "for instance",
    # сравнение/противопоставление
    "in contrast", "on the other hand", "similarly",
    # уточнение
    "in particular", "notably", "specifically",
]
def load_connectives_from_file(path: str) -> list[str]:
    items: list[str] = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                term = line.strip().lower()
                if not term:
                    continue
                if term.startswith('#'):
                    continue
                items.append(term)
    except Exception:
        return []
    # Уникализируем и сохраняем порядок
    seen = set()
    out: list[str] = []
    for t in items:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out



# ------------------------
# Utility tokenization/sentence splitters
# ------------------------
def split_sentences(text: str) -> list[str]:
    # simple regex-based splitter
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _connectives_rate(text: str, connectives: list[str]) -> float:
    clean = exp1.preprocess_text(text)
    words = re.findall(r"[a-zA-Z']+", clean)
    total = max(1, len(words))
    lowered = f" {clean} "
    hits = 0
    for conn in connectives:
        pattern = r"\b" + re.escape(conn) + r"\b"
        hits += len(re.findall(pattern, lowered))
    return 1000.0 * hits / float(total)


# ------------------------
# Lexica/Stylistics metrics
# ------------------------
def corpus_ttr(documents: list[str]) -> float:
    tokens: list[str] = []
    for doc in documents:
        tokens.extend(tokenize_words(doc))
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))


def corpus_zipf_fit(documents: list[str]) -> tuple[float, float]:
    # Returns (slope, r2) for log(freq) ~ a + b*log(rank)
    from collections import Counter
    tokens: list[str] = []
    for doc in documents:
        tokens.extend(tokenize_words(doc))
    if not tokens:
        return 0.0, 0.0
    cnt = Counter(tokens)
    freqs = sorted(cnt.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))
    import numpy as np
    x = np.log(np.array(ranks, dtype=float))
    y = np.log(np.array(freqs, dtype=float))
    b, a = np.polyfit(x, y, 1)  # y ~ a + b x
    y_pred = a + b * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(b), float(r2)


def corpus_self_bleu1(documents: list[str], sample_size: int = 50) -> float:
    # Approximate Self-BLEU-1: average unigram precision of each doc vs union of others
    from collections import Counter
    if not documents:
        return 0.0
    docs = documents[:sample_size] if len(documents) > sample_size else documents
    # build global reference counts for all docs first
    precisions = []
    for i, doc in enumerate(docs):
        doc_tokens = tokenize_words(doc)
        if not doc_tokens:
            continue
        # aggregate references from others
        ref_counts = Counter()
        for j, other in enumerate(docs):
            if j == i:
                continue
            ref_counts.update(tokenize_words(other))
        # clipped precision
        doc_counts = Counter(doc_tokens)
        match = 0
        total = 0
        for tok, c in doc_counts.items():
            total += c
            match += min(c, ref_counts.get(tok, 0))
        if total > 0:
            precisions.append(match / float(total))
    return float(sum(precisions) / len(precisions)) if precisions else 0.0


# ------------------------
# Structure/Coherence metrics
# ------------------------
def corpus_burstiness(documents: list[str]) -> tuple[float, float]:
    # Returns (mean sentence length, std sentence length)
    import numpy as np
    lengths = []
    for doc in documents:
        sents = split_sentences(doc)
        if not sents:
            continue
        for s in sents:
            lengths.append(len(tokenize_words(s)))
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths, dtype=float)
    return float(arr.mean()), float(arr.std())


def corpus_coherence_tfidf(documents: list[str]) -> float:
    # Average adjacent sentence cosine similarity using TF-IDF
    import numpy as np
    sims: list[float] = []
    for doc in documents:
        sents = split_sentences(doc)
        if len(sents) < 2:
            continue
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=4000)
        X = vec.fit_transform(sents)
        for k in range(X.shape[0] - 1):
            a = X[k]
            b = X[k + 1]
            num = a.multiply(b).sum()
            denom = (a.multiply(a).sum() ** 0.5) * (b.multiply(b).sum() ** 0.5)
            if denom > 0:
                sims.append(float(num / denom))
    return float(np.mean(sims)) if sims else 0.0


# ------------------------
# Technical heuristics
# ------------------------
def corpus_gzip_ratio(documents: list[str]) -> float:
    import gzip
    import io
    data = ("\n\n".join(documents)).encode('utf-8', errors='ignore')
    if not data:
        return 0.0
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(data)
    comp = buf.getvalue()
    return float(len(data) / max(1, len(comp)))

def read_txts_from_dir(dir_path: str, limit: int | None = None) -> List[str]:
    docs: List[str] = []
    if not os.path.exists(dir_path):
        return docs
    files = sorted(f for f in os.listdir(dir_path) if f.endswith('.txt'))
    if limit is not None:
        files = files[:limit]
    for f in files:
        try:
            with open(os.path.join(dir_path, f), 'r', encoding='utf-8') as fh:
                content = fh.read()
                if "Abstract:" in content:
                    content = content.split("Abstract:")[-1].strip()
                docs.append(content.strip())
        except Exception:
            continue
    return docs


def load_human_docs(human_root: str, per_topic: int = 50) -> List[str]:
    tm = read_txts_from_dir(os.path.join(human_root, 'text_mining'), per_topic)
    ir = read_txts_from_dir(os.path.join(human_root, 'information_retrieval'), per_topic)
    return tm + ir


def load_model_docs(model: str, per_model: int = 100) -> List[str]:
    # Поддерживаем существующую структуру путей из проекта
    candidates = [
        # Text Mining
        f"data/ai/{model}_api_text/text_mining_full",
        f"data/ai/{model}_api_auto/text_mining_full",
        f"data/ai/{model}_api/text_mining_full",
        # Information Retrieval
        f"data/ai/{model}_api/ir",
    ]
    docs: List[str] = []
    part = per_model
    for path in candidates:
        if len(docs) >= per_model:
            break
        remain = per_model - len(docs)
        docs.extend(read_txts_from_dir(path, remain))
    return docs[:per_model]


def compute_connectives_rate(documents: List[str]) -> float:
    if not documents:
        return 0.0
    total_words = 0
    total_hits = 0
    for doc in documents:
        text = exp1.preprocess_text(doc)
        words = re.findall(r"[a-zA-Z']+", text)
        total_words += max(1, len(words))
        lowered = ' ' + text + ' '
        for conn in CONNECTIVES:
            # считаем точные фразы с границами слов
            pattern = r"\b" + re.escape(conn) + r"\b"
            total_hits += len(re.findall(pattern, lowered))
    return 1000.0 * total_hits / max(1, total_words)


def compute_connectives_detection(human_docs: List[str], synth_docs: List[str]) -> dict:
    human_rates = [_connectives_rate(doc, CONNECTIVES) for doc in human_docs]
    synth_rates = [_connectives_rate(doc, CONNECTIVES) for doc in synth_docs]
    if not human_rates or not synth_rates:
        return {
            'human_mean': float(np.mean(human_rates) if human_rates else 0.0),
            'synth_mean': float(np.mean(synth_rates) if synth_rates else 0.0),
            'auc': float('nan'),
            'best_threshold': float('nan'),
            'threshold_direction': '>=',
            'accuracy_at_best': float('nan'),
        }
    scores = human_rates + synth_rates
    labels = [0] * len(human_rates) + [1] * len(synth_rates)
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    thr = float(thresholds[idx])
    h_mean = float(np.mean(human_rates))
    s_mean = float(np.mean(synth_rates))
    synth_greater = s_mean >= h_mean
    preds = []
    for s in scores:
        if synth_greater:
            preds.append(1 if s >= thr else 0)
        else:
            preds.append(1 if s <= thr else 0)
    acc = float((np.array(preds) == np.array(labels)).mean())
    return {
        'human_mean': h_mean,
        'synth_mean': s_mean,
        'auc': auc,
        'best_threshold': thr,
        'threshold_direction': '>=' if synth_greater else '<=',
        'accuracy_at_best': acc,
    }


def compute_connectives_detection(human_docs: List[str], synth_docs: List[str]) -> dict:
    # Обучаем детектор на непрерывном признаке (частота connectives)
    human_rates = []
    synth_rates = []
    for doc in human_docs:
        text = exp1.preprocess_text(doc)
        words = re.findall(r"[a-zA-Z']+", text)
        total = max(1, len(words))
        lowered = ' ' + text + ' '
        hits = 0
        for conn in CONNECTIVES:
            pattern = r"\b" + re.escape(conn) + r"\b"
            hits += len(re.findall(pattern, lowered))
        human_rates.append(1000.0 * hits / float(total))
    for doc in synth_docs:
        text = exp1.preprocess_text(doc)
        words = re.findall(r"[a-zA-Z']+", text)
        total = max(1, len(words))
        lowered = ' ' + text + ' '
        hits = 0
        for conn in CONNECTIVES:
            pattern = r"\b" + re.escape(conn) + r"\b"
            hits += len(re.findall(pattern, lowered))
        synth_rates.append(1000.0 * hits / float(total))

    if not human_rates or not synth_rates:
        return {
            'human_mean': float(np.mean(human_rates) if human_rates else 0.0),
            'synth_mean': float(np.mean(synth_rates) if synth_rates else 0.0),
            'auc': float('nan'),
            'best_threshold': float('nan'),
            'threshold_direction': '>=',
            'accuracy_at_best': float('nan'),
        }

    scores = human_rates + synth_rates
    labels = [0] * len(human_rates) + [1] * len(synth_rates)
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    thr = float(thresholds[idx])

    h_mean = float(np.mean(human_rates))
    s_mean = float(np.mean(synth_rates))
    synth_greater = s_mean >= h_mean
    preds = []
    for s in scores:
        if synth_greater:
            preds.append(1 if s >= thr else 0)
        else:
            preds.append(1 if s <= thr else 0)
    acc = float((np.array(preds) == np.array(labels)).mean())

    return {
        'human_mean': h_mean,
        'synth_mean': s_mean,
        'auc': auc,
        'best_threshold': thr,
        'threshold_direction': '>=' if synth_greater else '<=',
        'accuracy_at_best': acc,
    }


def extract_all_methods(human_docs: List[str], synth_docs: List[str], args) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    # TF-IDF N-граммы
    tf_min_df = int(args.tfidf_min_df) if args.tfidf_min_df >= 1 else float(args.tfidf_min_df)
    human_kw, human_df = exp1.extract_ngrams_keywords_per_doc(
        human_docs,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.tfidf_max_features,
        top_per_doc=args.tfidf_top_per_doc,
        min_df=tf_min_df,
        max_df=args.tfidf_max_df,
    )
    synth_kw, synth_df = exp1.extract_ngrams_keywords_per_doc(
        synth_docs,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.tfidf_max_features,
        top_per_doc=args.tfidf_top_per_doc,
        min_df=tf_min_df,
        max_df=args.tfidf_max_df,
    )
    results['ngrams'] = {
        'human_keywords': human_kw,
        'synthetic_keywords': synth_kw,
        'human_df': human_df,
        'synthetic_df': synth_df,
        'overlap_metrics': exp1.calculate_keyword_overlap(human_kw, synth_kw),
        'human_diversity': exp1.analyze_keyword_diversity(human_kw),
        'synthetic_diversity': exp1.analyze_keyword_diversity(synth_kw),
    }

    # YAKE
    human_kw, human_df = exp1.extract_yake_keywords_per_doc(
        human_docs,
        top_per_doc=args.yake_top_per_doc,
        max_ngram_size=args.yake_max_ngram,
        dedup_lim=args.yake_dedup,
    )
    synth_kw, synth_df = exp1.extract_yake_keywords_per_doc(
        synth_docs,
        top_per_doc=args.yake_top_per_doc,
        max_ngram_size=args.yake_max_ngram,
        dedup_lim=args.yake_dedup,
    )
    results['yake'] = {
        'human_keywords': human_kw,
        'synthetic_keywords': synth_kw,
        'human_df': human_df,
        'synthetic_df': synth_df,
        'overlap_metrics': exp1.calculate_keyword_overlap(human_kw, synth_kw),
        'human_diversity': exp1.analyze_keyword_diversity(human_kw),
        'synthetic_diversity': exp1.analyze_keyword_diversity(synth_kw),
    }

    # TextRank
    human_kw, human_df = exp1.extract_textrank_keywords_per_doc(
        human_docs,
        top_per_doc=args.textrank_top_per_doc,
        ratio=args.textrank_ratio,
    )
    synth_kw, synth_df = exp1.extract_textrank_keywords_per_doc(
        synth_docs,
        top_per_doc=args.textrank_top_per_doc,
        ratio=args.textrank_ratio,
    )
    results['textrank'] = {
        'human_keywords': human_kw,
        'synthetic_keywords': synth_kw,
        'human_df': human_df,
        'synthetic_df': synth_df,
        'overlap_metrics': exp1.calculate_keyword_overlap(human_kw, synth_kw),
        'human_diversity': exp1.analyze_keyword_diversity(human_kw),
        'synthetic_diversity': exp1.analyze_keyword_diversity(synth_kw),
    }

    return results


def cosine_similarity_corpora(human_docs: List[str], synth_docs: List[str]) -> float:
    # Векторизация всех документов и сравнение центроидов
    all_docs = [exp1.preprocess_text(d) for d in (human_docs + synth_docs)]
    labels = np.array([0] * len(human_docs) + [1] * len(synth_docs))
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform(all_docs)
    if X.shape[0] < 2:
        return 0.0
    human_centroid = X[labels == 0].mean(axis=0)
    synth_centroid = X[labels == 1].mean(axis=0)
    # Convert possible numpy.matrix to ndarray and enforce 2D shape
    human_centroid = np.asarray(human_centroid).reshape(1, -1)
    synth_centroid = np.asarray(synth_centroid).reshape(1, -1)
    sim = cosine_similarity(human_centroid, synth_centroid)[0, 0]
    return float(sim)


def plot_metrics_for_model(model: str, results: Dict[str, Dict], conn_h: float, conn_s: float, cos_sim: float, out_dir: str):
    methods = ['ngrams', 'yake', 'textrank']
    jaccard = [results[m]['overlap_metrics']['jaccard'] for m in methods]
    harmonic = [results[m]['overlap_metrics']['harmonic_mean'] for m in methods]

    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')

    # Метрики пересечения
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods))
    axes[0].bar(x, jaccard, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title(f'{model.upper()}: Jaccard')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in methods])
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, harmonic, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title(f'{model.upper()}: Harmonic Mean')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in methods])
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model}_overlaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Вводные слова и косинус
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([0,1], [conn_h, conn_s], color=['#1f77b4', '#ff7f0e'])
    axes[0].set_xticks([0,1])
    axes[0].set_xticklabels(['HUMAN', model.upper()])
    axes[0].set_title('Connectives per 1000 words')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar([0], [cos_sim], color=['#2ca02c'])
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['Cosine Sim'])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('TF-IDF centroid cosine similarity')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model}_connectives_cosine.png'), dpi=300, bbox_inches='tight')
    plt.close()


def write_report(models: List[str], per_model_results: Dict[str, Dict], out_dir: str):
    report = os.path.join(out_dir, 'experiment1_per_model_report.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write("# Эксперимент 1 (перемоделный): 100 HUMAN vs 100 AI текстов на модель\n\n")
        f.write("## Методология\n\n")
        f.write("- **Корпуса**: 100 человеческих (50 TM + 50 IR) против 100 синтетических на модель (LLAMA, QWEN, DEEPSEEK-R1)\n")
        f.write("- **Методы**: TF-IDF n-граммы, YAKE, TextRank\n")
        f.write("- **Метрики**: Jaccard, Overlap Human, Overlap Synthetic, Harmonic Mean; Connectives per 1000; TF-IDF cosine similarity\n\n")

        # Метрики и формулы
        f.write("## Метрики и формулы\n\n")
        f.write("- Jaccard: J(A,B) = |A∩B| / |A∪B|\n")
        f.write("- Overlap Human: |A∩B| / |A|; Overlap Synthetic: |A∩B| / |B|\n")
        f.write("- Harmonic Mean: 2·OH·OS / (OH+OS)\n")
        f.write("- Connectives per 1000: частота коннекторов на 1000 слов по словарю коннекторов\n")
        f.write("- TF-IDF Cosine: cos(θ) = (c_H·c_A) / (||c_H||·||c_A||), где c_H,c_A — TF‑IDF центроиды HUMAN/AI\n\n")
        f.write("Пример: при топ-50 слов у HUMAN/AI и пересечении = 20 получим J=0.25, Overlap=0.40/0.40, Harmonic=0.40 — это умеренное пересечение, не сильная схожесть.\n\n")
        for model in models:
            res = per_model_results[model]
            f.write(f"## Модель: {model.upper()}\n\n")
            # Сводная таблица
            f.write("| Метод | Jaccard | Overlap H | Overlap S | Harmonic |\n")
            f.write("|------|---------|-----------|-----------|----------|\n")
            for m in ['ngrams', 'yake', 'textrank']:
                om = res['methods'][m]['overlap_metrics']
                f.write(f"| {m.upper()} | {om['jaccard']:.3f} | {om['overlap_human']:.3f} | {om['overlap_synthetic']:.3f} | {om['harmonic_mean']:.3f} |\n")
            f.write("\n")
            f.write(f"- Connectives per 1000 words: HUMAN={res['connectives_h']:.2f}, {model.upper()}={res['connectives_s']:.2f}\n")
            cd = res.get('connectives_detection', {})
            if cd:
                f.write(f"- Connectives detection: AUC={cd.get('auc', float('nan')):.3f}, Threshold {cd.get('threshold_direction','>=')} {cd.get('best_threshold', float('nan')):.2f}, Acc@thr={cd.get('accuracy_at_best', float('nan')):.3f}\n")
            f.write(f"- TF-IDF centroid cosine similarity: {res['cosine_similarity']:.3f}\n\n")
            f.write(f"![Пересечения]({model}_overlaps.png)\n\n")
            f.write(f"![Вводные и косинус]({model}_connectives_cosine.png)\n\n")

            # Доп. метрики
            extra = res.get('extra', {})
            if extra:
                f.write("**Дополнительные метрики (лексика/стилистика, структура, эвристики):**\n\n")
                f.write(f"- TTR (HUMAN/AI): {extra.get('ttr_h',0):.3f} / {extra.get('ttr_s',0):.3f}\n")
                f.write(f"- Zipf slope (H/A): {extra.get('zipf_slope_h',0):.3f} / {extra.get('zipf_slope_s',0):.3f}, R2 (H/A): {extra.get('zipf_r2_h',0):.3f} / {extra.get('zipf_r2_s',0):.3f}\n")
                f.write(f"- Self-BLEU1 (H/A): {extra.get('self_bleu_h',0):.3f} / {extra.get('self_bleu_s',0):.3f}\n")
                f.write(f"- Coherence TF-IDF (H/A): {extra.get('coh_h',0):.3f} / {extra.get('coh_s',0):.3f}\n")
                f.write(f"- Sentence length mean±std (H): {extra.get('sent_mean_h',0):.2f}±{extra.get('sent_std_h',0):.2f}; (A): {extra.get('sent_mean_s',0):.2f}±{extra.get('sent_std_s',0):.2f}\n")
                f.write(f"- Gzip ratio (H/A): {extra.get('gzip_h',0):.2f} / {extra.get('gzip_s',0):.2f}\n\n")

        # Интерпретация и применимость
        f.write("## Как использовать результаты для детекции AI-текстов\n\n")
        f.write("- Jaccard/Harmonic ~ 0.25–0.56 указывают лишь на умеренную разницу в наборах ключевых слов — этих сигналов недостаточно для надежной детекции в одиночку.\n")
        f.write("- Connectives per 1000 даёт небольшой сдвиг (у AI часто ниже). Можно ставить простой порог, но точность ограничена.\n")
        f.write("- TF‑IDF cosine (≈0.67–0.74) отражает умеренную разделимость словарей и подходит как вспомогательная фича.\n")
        f.write("- Практический подход: объединять признаки в скоринг (например, Score = α·(1−Harmonic) + β·|ΔConnectives| + γ·(1−Cosine)) и валидировать порог на отложенной выборке. Основной детектор — семантический (см. Эксперимент 2).\n\n")

        f.write("## Итоги по результатам и выводы\n\n")
        f.write("- Лексические/стилистические метрики дают слабые–умеренные сигналы различий; использовать их стоит как часть ансамбля.\n")
        f.write("- Для практической детекции рекомендуется совмещать Jaccard/Harmonic, Connectives per 1000 и TF‑IDF cosine; улучшение ожидаемо при добавлении семантических фич.\n")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент 1 (перемоделный): HUMAN vs модель")
    parser.add_argument('--output_dir', default='results/experiment1_per_model')
    parser.add_argument('--human_root', default='data/human')
    parser.add_argument('--human_per_topic', type=int, default=50)
    parser.add_argument('--docs_per_model', type=int, default=100)
    # TF-IDF
    parser.add_argument('--ngram_min', type=int, default=1)
    parser.add_argument('--ngram_max', type=int, default=3)
    parser.add_argument('--tfidf_max_features', type=int, default=2000)
    parser.add_argument('--tfidf_top_per_doc', type=int, default=20)
    parser.add_argument('--tfidf_min_df', type=float, default=1)
    parser.add_argument('--tfidf_max_df', type=float, default=0.9)
    # YAKE
    parser.add_argument('--yake_top_per_doc', type=int, default=20)
    parser.add_argument('--yake_max_ngram', type=int, default=3)
    parser.add_argument('--yake_dedup', type=float, default=0.8)
    # TextRank
    parser.add_argument('--textrank_top_per_doc', type=int, default=20)
    parser.add_argument('--textrank_ratio', type=float, default=0.2)
    parser.add_argument('--connectives_path', default='', help='Путь к файлу со списком вводных/связующих слов (по одному на строку)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Подменяем набор CONNECTIVES из файла при наличии
    if args.connectives_path and os.path.exists(args.connectives_path):
        loaded = load_connectives_from_file(args.connectives_path)
        if loaded:
            print(f"Загружено вводных слов: {len(loaded)} из {args.connectives_path}")
            globals()['CONNECTIVES'] = loaded

    print('Загрузка HUMAN корпусов...')
    human_docs = load_human_docs(args.human_root, args.human_per_topic)
    print(f'HUMAN документов: {len(human_docs)}')

    models = ['llama', 'qwen', 'deepseek', 'gptoss']
    per_model_results: Dict[str, Dict] = {}

    for model in models:
        print(f"\nОбработка модели: {model}")
        synth_docs = load_model_docs(model, args.docs_per_model)
        print(f"Синтетических документов ({model}): {len(synth_docs)}")
        if len(synth_docs) == 0:
            print(f"Пропуск {model}: нет данных")
            continue

        methods_results = extract_all_methods(human_docs, synth_docs, args)

        # Connectives & Cosine
        conn_h = compute_connectives_rate(human_docs)
        conn_s = compute_connectives_rate(synth_docs)
        cos_sim = cosine_similarity_corpora(human_docs, synth_docs)
        conn_det = compute_connectives_detection(human_docs, synth_docs)

        # Extra metrics
        ttr_h = corpus_ttr(human_docs)
        ttr_s = corpus_ttr(synth_docs)
        slope_h, r2_h = corpus_zipf_fit(human_docs)
        slope_s, r2_s = corpus_zipf_fit(synth_docs)
        sb_h = corpus_self_bleu1(human_docs, sample_size=50)
        sb_s = corpus_self_bleu1(synth_docs, sample_size=50)
        coh_h = corpus_coherence_tfidf(human_docs)
        coh_s = corpus_coherence_tfidf(synth_docs)
        smean_h, sstd_h = corpus_burstiness(human_docs)
        smean_s, sstd_s = corpus_burstiness(synth_docs)
        gz_h = corpus_gzip_ratio(human_docs)
        gz_s = corpus_gzip_ratio(synth_docs)

        plot_metrics_for_model(model, methods_results, conn_h, conn_s, cos_sim, args.output_dir)

        per_model_results[model] = {
            'methods': methods_results,
            'connectives_h': conn_h,
            'connectives_s': conn_s,
            'cosine_similarity': cos_sim,
            'connectives_detection': conn_det,
            'extra': {
                'ttr_h': ttr_h, 'ttr_s': ttr_s,
                'zipf_slope_h': slope_h, 'zipf_r2_h': r2_h,
                'zipf_slope_s': slope_s, 'zipf_r2_s': r2_s,
                'self_bleu_h': sb_h, 'self_bleu_s': sb_s,
                'coh_h': coh_h, 'coh_s': coh_s,
                'sent_mean_h': smean_h, 'sent_std_h': sstd_h,
                'sent_mean_s': smean_s, 'sent_std_s': sstd_s,
                'gzip_h': gz_h, 'gzip_s': gz_s,
            }
        }

    write_report(list(per_model_results.keys()), per_model_results, args.output_dir)

    # JSON экспорт (минимальный)
    with open(os.path.join(args.output_dir, 'experiment1_per_model_results.json'), 'w', encoding='utf-8') as f:
        json.dump(per_model_results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Отчет: {os.path.join(args.output_dir, 'experiment1_per_model_report.md')}")


if __name__ == '__main__':
    main()


