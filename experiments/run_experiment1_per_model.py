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
            f.write(f"- TF-IDF centroid cosine similarity: {res['cosine_similarity']:.3f}\n\n")
            f.write(f"![Пересечения]({model}_overlaps.png)\n\n")
            f.write(f"![Вводные и косинус]({model}_connectives_cosine.png)\n\n")

        # Интерпретация и применимость
        f.write("## Как использовать результаты для детекции AI-текстов\n\n")
        f.write("- Низкие значения Jaccard/Harmonic указывают на различия в лексике и ключевых фразах между HUMAN и AI; это сигнал для детекции.\n")
        f.write("- Connectives per 1000: переизбыток/недостаток связующих слов у AI относительно HUMAN позволяет построить простой линейный порог.\n")
        f.write("- TF-IDF cosine similarity между центроидами корпусов: чем ниже сходство, тем проще отделять AI от HUMAN на уровне словаря.\n")
        f.write("- Рекомендуется ансамбль из (TextRank Harmonic + Connectives gap + Cosine), что повышает устойчивость к перегенерациям.\n")


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
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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

        plot_metrics_for_model(model, methods_results, conn_h, conn_s, cos_sim, args.output_dir)

        per_model_results[model] = {
            'methods': methods_results,
            'connectives_h': conn_h,
            'connectives_s': conn_s,
            'cosine_similarity': cos_sim,
        }

    write_report(list(per_model_results.keys()), per_model_results, args.output_dir)

    # JSON экспорт (минимальный)
    with open(os.path.join(args.output_dir, 'experiment1_per_model_results.json'), 'w', encoding='utf-8') as f:
        json.dump(per_model_results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Отчет: {os.path.join(args.output_dir, 'experiment1_per_model_report.md')}")


if __name__ == '__main__':
    main()


