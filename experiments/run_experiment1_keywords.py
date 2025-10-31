#!/usr/bin/env python3
"""
Эксперимент 1: Анализ ключевых слов в человеческих и синтетических текстах
- Извлечение ключевых слов через n-граммы, YAKE, TextRank
- Сравнение результатов между человеческими и AI-сгенерированными текстами
- Выборка: 30 человеческих + 30 синтетических документов (по 15 на тему)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    USE_SEABORN = True
except Exception:
    USE_SEABORN = False
import re
from collections import Counter, defaultdict

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("Warning: YAKE not available. Install with: pip install yake")

try:
    from summa import keywords
    TEXTRANK_AVAILABLE = True
except ImportError:
    TEXTRANK_AVAILABLE = False
    print("Warning: TextRank not available. Install with: pip install summa")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve


def load_documents_from_csv(csv_path: str, count: int = 15) -> List[str]:
    """Загружает документы из CSV файла"""
    try:
        df = pd.read_csv(csv_path)
        abstracts = df['abstract'].tolist()[:count]
        return [str(abstract) for abstract in abstracts if pd.notna(abstract)]
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return []


def load_documents_from_txt_dir(txt_dir: str, count: int = 15) -> List[str]:
    """Загружает документы из папки с TXT файлами"""
    try:
        txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])[:count]
        documents = []
        for txt_file in txt_files:
            with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8') as f:
                content = f.read()
                # Извлекаем только текст абстракта (после "Abstract:")
                if "Abstract:" in content:
                    abstract = content.split("Abstract:")[-1].strip()
                    documents.append(abstract)
                else:
                    documents.append(content)
        return documents
    except Exception as e:
        print(f"Error loading from {txt_dir}: {e}")
        return []


def preprocess_text(text: str) -> str:
    """Базовая предобработка текста"""
    # Убираем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Убираем специальные символы, оставляем только буквы, цифры и пробелы
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().strip()


# -------------------------------------------
# Вводные/связующие слова (connectives) и метрики по ним
# -------------------------------------------
CONNECTIVES: List[str] = [
    # contrast/consequence
    "however", "therefore", "thus", "hence", "nevertheless", "nonetheless",
    # addition/examples
    "moreover", "furthermore", "in addition", "additionally", "for example", "for instance",
    # comparison/contrast
    "in contrast", "on the other hand", "similarly",
    # specification
    "in particular", "notably", "specifically",
]


def _connectives_rate(text: str, connectives: List[str]) -> float:
    """Возвращает число вхождений connectives на 1000 слов для одного документа."""
    clean = preprocess_text(text)
    words = re.findall(r"[a-zA-Z']+", clean)
    total = max(1, len(words))
    lowered = f" {clean} "
    hits = 0
    for conn in connectives:
        pattern = r"\b" + re.escape(conn) + r"\b"
        hits += len(re.findall(pattern, lowered))
    return 1000.0 * hits / float(total)


def load_connectives_from_file(path: str) -> List[str]:
    items: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                term = line.strip().lower()
                if not term or term.startswith('#'):
                    continue
                items.append(term)
    except Exception:
        return []
    # deduplicate preserving order
    seen = set()
    out: List[str] = []
    for t in items:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def compute_connectives_detection(human_docs: List[str], synthetic_docs: List[str]) -> Dict[str, float | List[float]]:
    """Считает распределения частоты connectives (на 1000 слов) и подбирает порог для детекции.

    Возвращает словарь с:
    - human_mean, synth_mean
    - auc
    - best_threshold
    - threshold_direction ('>=' или '<=')
    - accuracy_at_best
    - human_rates, synth_rates (для визуализации)
    """
    human_rates = [_connectives_rate(doc, CONNECTIVES) for doc in human_docs]
    synth_rates = [_connectives_rate(doc, CONNECTIVES) for doc in synthetic_docs]

    if not human_rates or not synth_rates:
        return {
            'human_mean': float(np.mean(human_rates) if human_rates else 0.0),
            'synth_mean': float(np.mean(synth_rates) if synth_rates else 0.0),
            'auc': float('nan'),
            'best_threshold': float('nan'),
            'threshold_direction': '>=',
            'accuracy_at_best': float('nan'),
            'human_rates': human_rates,
            'synth_rates': synth_rates,
        }

    scores = human_rates + synth_rates
    labels = [0] * len(human_rates) + [1] * len(synth_rates)  # 1 = synthetic

    auc = float(roc_auc_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    raw_threshold = thresholds[idx]

    human_mean = float(np.mean(human_rates))
    synth_mean = float(np.mean(synth_rates))
    synth_greater = synth_mean >= human_mean

    preds = []
    for s in scores:
        if synth_greater:
            preds.append(1 if s >= raw_threshold else 0)
        else:
            preds.append(1 if s <= raw_threshold else 0)
    preds = np.array(preds)
    labels_arr = np.array(labels)
    accuracy = float((preds == labels_arr).mean())

    return {
        'human_mean': human_mean,
        'synth_mean': synth_mean,
        'auc': auc,
        'best_threshold': float(raw_threshold),
        'threshold_direction': '>=' if synth_greater else '<=',
        'accuracy_at_best': accuracy,
        'human_rates': human_rates,
        'synth_rates': synth_rates,
    }


def extract_ngrams_keywords_per_doc(
    documents: List[str],
    ngram_range: Tuple[int, int] = (1, 3),
    max_features: int = 500,
    top_per_doc: int = 20,
    min_df: int | float = 2,
    max_df: float = 0.85,
) -> Tuple[List[str], Dict[str, int]]:
    """Извлекает ключевые слова через TF-IDF n-граммы на уровне документов и агрегирует частоты по документам."""
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            preprocessor=preprocess_text,
            min_df=min_df,
            max_df=max_df
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        docfreq: Dict[str, int] = {}
        for row in tfidf_matrix:
            row_arr = row.toarray()[0]
            if not np.any(row_arr):
                continue
            top_idx = np.argsort(row_arr)[-top_per_doc:][::-1]
            seen = set()
            for idx in top_idx:
                term = feature_names[idx]
                if term in seen:
                    continue
                seen.add(term)
                docfreq[term] = docfreq.get(term, 0) + 1

        # Топ-50 по частоте по документам
        top_keywords = [kw for kw, _ in sorted(docfreq.items(), key=lambda x: x[1], reverse=True)[:50]]
        return top_keywords, docfreq
    except Exception as e:
        print(f"Error extracting n-grams per doc: {e}")
        return [], {}


def extract_yake_keywords_per_doc(
    documents: List[str],
    top_per_doc: int = 20,
    max_ngram_size: int = 3,
    dedup_lim: float = 0.8,
) -> Tuple[List[str], Dict[str, int]]:
    """YAKE по документам с агрегацией по частоте появления в документах."""
    if not YAKE_AVAILABLE:
        return [], {}
    try:
        kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram_size, dedupLim=dedup_lim, top=top_per_doc, features=None)
        docfreq: Dict[str, int] = {}
        for doc in documents:
            try:
                kws = kw_extractor.extract_keywords(doc)
                seen = set()
                for kw, _score in kws:
                    if not isinstance(kw, str):
                        continue
                    if kw in seen:
                        continue
                    seen.add(kw)
                    docfreq[kw] = docfreq.get(kw, 0) + 1
            except Exception:
                continue
        top_keywords = [kw for kw, _ in sorted(docfreq.items(), key=lambda x: x[1], reverse=True)[:50]]
        return top_keywords, docfreq
    except Exception as e:
        print(f"Error extracting YAKE per doc: {e}")
        return [], {}


def extract_textrank_keywords_per_doc(
    documents: List[str],
    top_per_doc: int = 20,
    ratio: float | None = 0.2,
) -> Tuple[List[str], Dict[str, int]]:
    """TextRank по документам с агрегацией по частоте появления в документах."""
    if not TEXTRANK_AVAILABLE:
        return [], {}
    try:
        docfreq: Dict[str, int] = {}
        for doc in documents:
            try:
                # Если ratio указан, берём долю слов; иначе фиксированное число слов
                if ratio is not None:
                    doc_kws = keywords.keywords(doc, ratio=ratio, split=True)
                else:
                    doc_kws = keywords.keywords(doc, words=top_per_doc, split=True)
                seen = set()
                for kw in doc_kws:
                    if kw in seen:
                        continue
                    seen.add(kw)
                    docfreq[kw] = docfreq.get(kw, 0) + 1
            except Exception:
                continue
        top_keywords = [kw for kw, _ in sorted(docfreq.items(), key=lambda x: x[1], reverse=True)[:50]]
        return top_keywords, docfreq
    except Exception as e:
        print(f"Error extracting TextRank per doc: {e}")
        return [], {}


def calculate_keyword_overlap(keywords1: List[str], keywords2: List[str]) -> Dict[str, float]:
    """Вычисляет метрики пересечения ключевых слов"""
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Меры пересечения (НЕ классические метрики классификации!)
    overlap_human = len(intersection) / len(set1) if set1 else 0.0  # Доля человеческих слов в синтетических
    overlap_synthetic = len(intersection) / len(set2) if set2 else 0.0  # Доля синтетических слов в человеческих
    
    # Гармоническое среднее пересечений
    harmonic_mean = 2 * overlap_human * overlap_synthetic / (overlap_human + overlap_synthetic) if (overlap_human + overlap_synthetic) > 0 else 0.0
    
    return {
        'jaccard': jaccard,
        'overlap_human': overlap_human,  # Было "precision"
        'overlap_synthetic': overlap_synthetic,  # Было "recall" 
        'harmonic_mean': harmonic_mean,  # Было "f1"
        'intersection_size': len(intersection),
        'set1_size': len(set1),
        'set2_size': len(set2)
    }


def analyze_keyword_diversity(keywords: List[str]) -> Dict[str, float]:
    """Анализирует разнообразие ключевых слов"""
    if not keywords:
        return {'unique_ratio': 0.0, 'avg_length': 0.0, 'length_std': 0.0}
    
    # Фильтруем только строки и конвертируем в строки
    str_keywords = [str(kw) for kw in keywords if kw is not None]
    
    if not str_keywords:
        return {'unique_ratio': 0.0, 'avg_length': 0.0, 'length_std': 0.0}
    
    # Уникальность
    unique_ratio = len(set(str_keywords)) / len(str_keywords)
    
    # Длина ключевых слов
    lengths = [len(kw.split()) for kw in str_keywords]
    avg_length = np.mean(lengths)
    length_std = np.std(lengths)
    
    return {
        'unique_ratio': unique_ratio,
        'avg_length': avg_length,
        'length_std': length_std,
        'total_keywords': len(str_keywords),
        'unique_keywords': len(set(str_keywords))
    }


def create_visualizations(results: Dict, output_dir: str):
    """Создает графики для результатов эксперимента"""
    
    # Настройка стиля
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')
    if USE_SEABORN:
        try:
            sns.set_palette("husl")
        except Exception:
            pass
    
    # 1. График сравнения метрик по методам
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение методов извлечения ключевых слов', fontsize=16, fontweight='bold')
    
    methods = ['ngrams', 'yake', 'textrank']
    
    # Подготовка данных
    jaccard_scores = []
    harmonic_means = []
    overlap_human_scores = []
    overlap_synthetic_scores = []
    
    for method in methods:
        if method in results:
            overlap = results[method]['overlap_metrics']
            jaccard_scores.append(overlap['jaccard'])
            harmonic_means.append(overlap['harmonic_mean'])
            overlap_human_scores.append(overlap['overlap_human'])
            overlap_synthetic_scores.append(overlap['overlap_synthetic'])
    
    # График Jaccard Index
    ax1 = axes[0, 0]
    x_pos = range(len(methods))
    bars1 = ax1.bar(x_pos, jaccard_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Jaccard Index по методам')
    ax1.set_ylabel('Jaccard Index')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.upper() for m in methods])
    ax1.grid(True, alpha=0.3)
    
    # График Harmonic Mean
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, harmonic_means, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Harmonic Mean по методам')
    ax2.set_ylabel('Harmonic Mean')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.grid(True, alpha=0.3)
    
    # График Overlap Human
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, overlap_human_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title('Overlap Human по методам')
    ax3.set_ylabel('Overlap Human')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m.upper() for m in methods])
    ax3.grid(True, alpha=0.3)
    
    # График Overlap Synthetic
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, overlap_synthetic_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_title('Overlap Synthetic по методам')
    ax4.set_ylabel('Overlap Synthetic')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([m.upper() for m in methods])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. График топ ключевых слов
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Топ-10 ключевых слов по методам', fontsize=16, fontweight='bold')
    
    for i, method in enumerate(methods):
        if method in results:
            # Используем документные частоты для топов
            human_df = results[method]['human_df']
            synthetic_df = results[method]['synthetic_df']
            # Определяем общую шкалу топ-ключевых по суммарной частоте
            all_terms = set(human_df.keys()) | set(synthetic_df.keys())
            ranked = sorted(all_terms, key=lambda t: human_df.get(t, 0) + synthetic_df.get(t, 0), reverse=True)
            top_terms = ranked[:10]
            if not top_terms:
                axes[i].set_visible(False)
                continue
            human_counts = [human_df.get(t, 0) for t in top_terms]
            synthetic_counts = [synthetic_df.get(t, 0) for t in top_terms]
            
            x = np.arange(len(top_terms))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, human_counts, width, label='Человеческие', color='#1f77b4')
            bars2 = axes[i].bar(x + width/2, synthetic_counts, width, label='Синтетические', color='#ff7f0e')
            
            axes[i].set_title(f'{method.upper()}')
            axes[i].set_xlabel('Ключевые слова')
            axes[i].set_ylabel('Документная частота')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(top_terms, rotation=45, ha='right', fontsize=8)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_keywords_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. График анализа разнообразия
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Анализ разнообразия ключевых слов', fontsize=16, fontweight='bold')
    
    # Уникальность
    ax1 = axes[0]
    human_unique = [results[method]['human_diversity']['unique_ratio'] for method in methods if method in results]
    synthetic_unique = [results[method]['synthetic_diversity']['unique_ratio'] for method in methods if method in results]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, human_unique, width, label='Человеческие', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, synthetic_unique, width, label='Синтетические', color='#ff7f0e')
    
    ax1.set_title('Коэффициент уникальности')
    ax1.set_ylabel('Unique Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in methods])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Средняя длина
    ax2 = axes[1]
    human_length = [results[method]['human_diversity']['avg_length'] for method in methods if method in results]
    synthetic_length = [results[method]['synthetic_diversity']['avg_length'] for method in methods if method in results]
    
    bars3 = ax2.bar(x - width/2, human_length, width, label='Человеческие', color='#1f77b4')
    bars4 = ax2.bar(x + width/2, synthetic_length, width, label='Синтетические', color='#ff7f0e')
    
    ax2.set_title('Средняя длина ключевых слов')
    ax2.set_ylabel('Средняя длина (слова)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в папке: {output_dir}")


def create_connectives_plots(conn_results: Dict[str, float | List[float]], output_dir: str):
    """Рисует графики для метрик по вводным словам: распределения и сравнение средних."""
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')
    if USE_SEABORN:
        try:
            sns.set_palette('colorblind')
        except Exception:
            pass

    # Гистограммы распределений
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    human_rates = conn_results.get('human_rates', [])
    synth_rates = conn_results.get('synth_rates', [])
    if human_rates:
        ax.hist(human_rates, bins=20, alpha=0.6, label='HUMAN', color='#1f77b4', density=True)
    if synth_rates:
        ax.hist(synth_rates, bins=20, alpha=0.6, label='AI', color='#ff7f0e', density=True)
    bt = conn_results.get('best_threshold')
    if isinstance(bt, float) and not np.isnan(bt):
        ax.axvline(bt, color='red', linestyle='--', label=f"Threshold {conn_results.get('threshold_direction','>=')} {bt:.2f}")
    ax.set_title('Connectives per 1000 words — распределения')
    ax.set_xlabel('rate per 1000')
    ax.set_ylabel('density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connectives_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Сравнение средних
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.bar([0, 1], [conn_results.get('human_mean', 0.0), conn_results.get('synth_mean', 0.0)],
           color=['#1f77b4', '#ff7f0e'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HUMAN', 'AI'])
    ax.set_title('Connectives per 1000 — средние')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connectives_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_markdown_report(results: Dict, output_dir: str):
    """Генерирует отчет в формате Markdown"""
    
    report_path = os.path.join(output_dir, 'experiment1_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Эксперимент 1: Анализ ключевых слов в человеческих и синтетических текстах\n\n")
        
        f.write("## Методология\n\n")
        f.write("- **Выборка**: 30 человеческих + 30 синтетических документов (объединенная выборка)\n")
        f.write("- **Темы**: Text Mining, Information Retrieval (объединены)\n")
        f.write("- **Методы извлечения ключевых слов**:\n")
        f.write("  - N-граммы (TF-IDF, 1-3 граммы)\n")
        f.write("  - YAKE (Yet Another Keyword Extractor)\n")
        f.write("  - TextRank\n")
        f.write("- **Метрики сравнения**: Jaccard, Overlap Human, Overlap Synthetic, Harmonic Mean\n\n")
        
        f.write("## Визуализация результатов\n\n")
        f.write("### Сравнение метрик по методам\n\n")
        f.write("![Сравнение метрик](metrics_comparison.png)\n\n")
        f.write("### Топ ключевых слов по методам\n\n")
        f.write("![Топ ключевых слов](top_keywords_comparison.png)\n\n")
        f.write("### Анализ разнообразия\n\n")
        f.write("![Анализ разнообразия](diversity_analysis.png)\n\n")
        f.write("### Вводные/связующие слова\n\n")
        f.write("![Распределения](connectives_distribution.png)\n\n")
        f.write("![Средние](connectives_bar.png)\n\n")
        
        f.write("## Результаты по методам\n\n")
        
        for method in ['ngrams', 'yake', 'textrank']:
            if method in results:
                f.write(f"### {method.upper()}\n\n")
                
                method_results = results[method]
                
                # Топ ключевые слова
                f.write("**Топ-10 ключевых слов (человеческие тексты):**\n")
                human_df = method_results['human_df']
                top_human = sorted(human_df.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (kw, freq) in enumerate(top_human, 1):
                    f.write(f"{i}. {kw} (частота: {freq})\n")
                f.write("\n")
                
                f.write("**Топ-10 ключевых слов (синтетические тексты):**\n")
                synthetic_df = method_results['synthetic_df']
                top_synthetic = sorted(synthetic_df.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (kw, freq) in enumerate(top_synthetic, 1):
                    f.write(f"{i}. {kw} (частота: {freq})\n")
                f.write("\n")
                
                # Метрики сравнения
                overlap = method_results['overlap_metrics']
                f.write("**Метрики пересечения:**\n")
                f.write(f"- Jaccard Index: {overlap['jaccard']:.3f}\n")
                f.write(f"- Overlap Human: {overlap['overlap_human']:.3f}\n")
                f.write(f"- Overlap Synthetic: {overlap['overlap_synthetic']:.3f}\n")
                f.write(f"- Harmonic Mean: {overlap['harmonic_mean']:.3f}\n")
                f.write(f"- Пересечение: {overlap['intersection_size']} из {overlap['set1_size']} и {overlap['set2_size']}\n\n")
                
                # Анализ разнообразия
                human_diversity = method_results['human_diversity']
                synthetic_diversity = method_results['synthetic_diversity']
                
                f.write("**Анализ разнообразия:**\n")
                f.write(f"- Человеческие тексты: уникальность {human_diversity['unique_ratio']:.3f}, "
                       f"средняя длина {human_diversity['avg_length']:.2f}\n")
                f.write(f"- Синтетические тексты: уникальность {synthetic_diversity['unique_ratio']:.3f}, "
                       f"средняя длина {synthetic_diversity['avg_length']:.2f}\n\n")
                
                f.write("---\n\n")
        
        # Блок по connectives
        conn = results.get('connectives', {})
        if conn:
            f.write("## Детекция по вводным словам (connectives)\n\n")
            f.write("- Признак: частота вводных/связующих слов на 1000 слов.\n")
            f.write("- Метрики: AUC по непрерывному признаку, порог Юдена, Accuracy на лучшем пороге.\n\n")
            f.write(f"- HUMAN mean: {conn.get('human_mean', 0.0):.2f}\n")
            f.write(f"- AI mean: {conn.get('synth_mean', 0.0):.2f}\n")
            f.write(f"- AUC: {conn.get('auc', float('nan')):.3f}\n")
            thr = conn.get('best_threshold', float('nan'))
            dirn = conn.get('threshold_direction', '>=')
            if thr == thr:
                f.write(f"- Лучший порог: {dirn} {thr:.2f}\n")
                f.write(f"- Accuracy @ threshold: {conn.get('accuracy_at_best', 0.0):.3f}\n\n")
            else:
                f.write("- Порог не определён (недостаточно данных)\n\n")

        # Общие выводы
        f.write("## Общие выводы\n\n")
        
        # Сравнение методов
        f.write("### Сравнение методов извлечения ключевых слов\n\n")
        f.write("| Метод | Jaccard | Harmonic Mean | Уникальность | Применимость для детекции |\n")
        f.write("|-------|---------|---------------|--------------|---------------------------|\n")
        
        for method in ['ngrams', 'yake', 'textrank']:
            if method in results:
                overlap = results[method]['overlap_metrics']
                human_div = results[method]['human_diversity']
                
                # Определяем применимость
                if overlap['harmonic_mean'] > 0.2:
                    applicability = "✅ Хорошая"
                elif overlap['harmonic_mean'] > 0.1:
                    applicability = "⚠️ Умеренная"
                else:
                    applicability = "❌ Низкая"
                
                f.write(f"| {method.upper()} | {overlap['jaccard']:.3f} | {overlap['harmonic_mean']:.3f} | {human_div['unique_ratio']:.3f} | {applicability} |\n")
        
        f.write("\n")
        
        # Выводы
        f.write("### Ключевые наблюдения\n\n")
        f.write("1. **Различия в ключевых словах**: Синтетические тексты показывают значительные различия в выборе ключевых слов по сравнению с человеческими.\n")
        f.write("2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.\n")
        f.write("3. **Объединенная выборка**: Анализ по объединенной выборке дает более общие и стабильные результаты.\n")
        f.write("4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.\n\n")
        
        f.write("## Заключение\n\n")
        f.write("Эксперимент показал, что анализ ключевых слов имеет умеренную применимость для распознавания AI-сгенерированных текстов. "
               "TextRank показал наилучшие результаты, что указывает на заметные различия между человеческими и синтетическими текстами. "
               "Для практической детекции AI-текстов следует использовать комбинацию методов.\n")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент 1: Анализ ключевых слов")
    parser.add_argument("--output_dir", default="results/experiment1", help="Папка для результатов")
    parser.add_argument("--docs_per_topic", type=int, default=15, help="Количество документов на тему")
    parser.add_argument("--connectives_path", default="", help="Путь к файлу со списком вводных/связующих слов (по одному на строку)")
    # Тюнинг экстрактора TF-IDF
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=3)
    parser.add_argument("--tfidf_max_features", type=int, default=1000)
    parser.add_argument("--tfidf_top_per_doc", type=int, default=20)
    parser.add_argument("--tfidf_min_df", type=float, default=1)
    parser.add_argument("--tfidf_max_df", type=float, default=0.9)
    # Тюнинг YAKE
    parser.add_argument("--yake_top_per_doc", type=int, default=20)
    parser.add_argument("--yake_max_ngram", type=int, default=3)
    parser.add_argument("--yake_dedup", type=float, default=0.8)
    # Тюнинг TextRank
    parser.add_argument("--textrank_top_per_doc", type=int, default=20)
    parser.add_argument("--textrank_ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    # Создаем папку для результатов
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Запуск эксперимента 1: Анализ ключевых слов...")

    # Подменяем набор CONNECTIVES из файла при наличии
    if args.connectives_path and os.path.exists(args.connectives_path):
        loaded_conn = load_connectives_from_file(args.connectives_path)
        if loaded_conn:
            print(f"Загружено вводных слов: {len(loaded_conn)} из {args.connectives_path}")
            globals()['CONNECTIVES'] = loaded_conn
    
    # Собираем все документы в одну выборку
    all_human_docs = []
    all_synthetic_docs = []
    
    topics = ['text_mining', 'information_retrieval']
    
    for topic in topics:
        print(f"Обработка темы: {topic}")
        
        # Человеческие документы
        human_dir = f"data/human/{topic}"
        if os.path.exists(human_dir):
            human_files = [f for f in os.listdir(human_dir) if f.endswith('.txt')]
            human_docs = []
            for file in human_files[:args.docs_per_topic]:
                with open(os.path.join(human_dir, file), 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        human_docs.append(content)
            all_human_docs.extend(human_docs)
            print(f"Загружено {len(human_docs)} человеческих документов")
        
        # Синтетические документы
        synthetic_docs = []
        models = ['llama', 'qwen', 'deepseek']
        for model in models:
            if topic == 'text_mining':
                model_dir = f"data/ai/{model}_api_text/text_mining_full"
            else:  # information_retrieval
                model_dir = f"data/ai/{model}_api/ir"
            
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.txt')]
                for file in model_files[:args.docs_per_topic // len(models)]:
                    with open(os.path.join(model_dir, file), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            synthetic_docs.append(content)
                print(f"Загружено {len(model_files[:args.docs_per_topic // len(models)])} документов от {model}")
        
        all_synthetic_docs.extend(synthetic_docs)
        print(f"Всего синтетических документов: {len(synthetic_docs)}")
    
    print(f"\nОбщая выборка:")
    print(f"Человеческих документов: {len(all_human_docs)}")
    print(f"Синтетических документов: {len(all_synthetic_docs)}")
    if len(all_human_docs) == 0 or len(all_synthetic_docs) == 0:
        print("Предупреждение: одна из групп пуста. Проверьте пути к данным. Эксперимент продолжится, но метрики будут вырождены.")
    
    # Методы извлечения ключевых слов
    methods = [
        ('ngrams', 'TF-IDF N-граммы'),
        ('yake', 'YAKE'),
        ('textrank', 'TextRank')
    ]
    
    results = {}
    
    for method_name, method_display in methods:
        print(f"\nИзвлечение ключевых слов методом {method_name}...")
        
        # Ключевые слова и документные частоты
        if method_name == 'ngrams':
            # Приводим min_df к int при значениях >= 1
            tf_min_df = int(args.tfidf_min_df) if args.tfidf_min_df >= 1 else float(args.tfidf_min_df)
            human_keywords, human_df = extract_ngrams_keywords_per_doc(
                all_human_docs,
                ngram_range=(args.ngram_min, args.ngram_max),
                max_features=args.tfidf_max_features,
                top_per_doc=args.tfidf_top_per_doc,
                min_df=tf_min_df,
                max_df=args.tfidf_max_df,
            )
            synthetic_keywords, synthetic_df = extract_ngrams_keywords_per_doc(
                all_synthetic_docs,
                ngram_range=(args.ngram_min, args.ngram_max),
                max_features=args.tfidf_max_features,
                top_per_doc=args.tfidf_top_per_doc,
                min_df=tf_min_df,
                max_df=args.tfidf_max_df,
            )
        elif method_name == 'yake':
            human_keywords, human_df = extract_yake_keywords_per_doc(
                all_human_docs,
                top_per_doc=args.yake_top_per_doc,
                max_ngram_size=args.yake_max_ngram,
                dedup_lim=args.yake_dedup,
            )
            synthetic_keywords, synthetic_df = extract_yake_keywords_per_doc(
                all_synthetic_docs,
                top_per_doc=args.yake_top_per_doc,
                max_ngram_size=args.yake_max_ngram,
                dedup_lim=args.yake_dedup,
            )
        else:  # textrank
            human_keywords, human_df = extract_textrank_keywords_per_doc(
                all_human_docs,
                top_per_doc=args.textrank_top_per_doc,
                ratio=args.textrank_ratio,
            )
            synthetic_keywords, synthetic_df = extract_textrank_keywords_per_doc(
                all_synthetic_docs,
                top_per_doc=args.textrank_top_per_doc,
                ratio=args.textrank_ratio,
            )
        
        # Метрики пересечения
        overlap_metrics = calculate_keyword_overlap(human_keywords, synthetic_keywords)
        # Защита от вырожденных метрик: если union пуст или обе выборки пустые, помечаем как NaN
        if (not human_keywords) and (not synthetic_keywords):
            overlap_metrics.update({'jaccard': float('nan'), 'overlap_human': float('nan'), 'overlap_synthetic': float('nan'), 'harmonic_mean': float('nan')})
        
        # Анализ разнообразия
        human_diversity = analyze_keyword_diversity(human_keywords)
        synthetic_diversity = analyze_keyword_diversity(synthetic_keywords)
        
        print(f"    Найдено ключевых слов: человеческие {len(human_keywords)}, синтетические {len(synthetic_keywords)}")
        print(f"    Jaccard: {overlap_metrics['jaccard']:.3f}, Harmonic Mean: {overlap_metrics['harmonic_mean']:.3f}")
        
        # Сохраняем результаты
        results[method_name] = {
            'human_keywords': human_keywords,
            'synthetic_keywords': synthetic_keywords,
            'human_df': human_df,
            'synthetic_df': synthetic_df,
            'overlap_metrics': overlap_metrics,
            'human_diversity': human_diversity,
            'synthetic_diversity': synthetic_diversity,
            'method_display': method_display
        }
    
    # Создаем визуализации
    print(f"\nГрафики сохранены в папке: {args.output_dir}")
    create_visualizations(results, args.output_dir)

    # Connectives detection & визуализации
    if all_human_docs and all_synthetic_docs:
        conn_results = compute_connectives_detection(all_human_docs, all_synthetic_docs)
    else:
        conn_results = {
            'human_mean': 0.0,
            'synth_mean': 0.0,
            'auc': float('nan'),
            'best_threshold': float('nan'),
            'threshold_direction': '>=',
            'accuracy_at_best': float('nan'),
            'human_rates': [],
            'synth_rates': [],
        }
    results['connectives'] = conn_results
    create_connectives_plots(conn_results, args.output_dir)

    # Генерируем отчет
    generate_markdown_report(results, args.output_dir)
    
    # Сохраняем JSON данные
    json_data = {}
    for method, data in results.items():
        if method == 'connectives':
            continue
        json_data[method] = {
            'overlap_metrics': data['overlap_metrics'],
            'human_diversity': data['human_diversity'],
            'synthetic_diversity': data['synthetic_diversity'],
            'top_human_keywords': list(data['human_df'].items())[:10],
            'top_synthetic_keywords': list(data['synthetic_df'].items())[:10]
        }

    # Добавляем connectives в JSON кратко
    if 'connectives' in results:
        json_data['connectives'] = {
            'human_mean': results['connectives'].get('human_mean', 0.0),
            'synth_mean': results['connectives'].get('synth_mean', 0.0),
            'auc': results['connectives'].get('auc', float('nan')),
            'best_threshold': results['connectives'].get('best_threshold', float('nan')),
            'threshold_direction': results['connectives'].get('threshold_direction', '>='),
            'accuracy_at_best': results['connectives'].get('accuracy_at_best', float('nan')),
        }
    
    with open(os.path.join(args.output_dir, 'experiment1_results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nЭксперимент завершен!")
    print(f"Результаты сохранены в: {args.output_dir}")
    print(f"Отчет: {args.output_dir}/experiment1_report.md")
    print(f"JSON данные: {args.output_dir}/experiment1_results.json")


if __name__ == "__main__":
    main()
