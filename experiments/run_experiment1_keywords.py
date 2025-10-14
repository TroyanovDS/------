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
import seaborn as sns
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


def extract_ngrams_keywords(documents: List[str], ngram_range: Tuple[int, int] = (1, 3), max_features: int = 100) -> List[str]:
    """Извлекает ключевые слова через TF-IDF n-граммы"""
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            preprocessor=preprocess_text
        )
        
        # Объединяем все документы для обучения TF-IDF
        all_text = ' '.join(documents)
        tfidf_matrix = vectorizer.fit_transform([all_text])
        
        # Получаем топ ключевых слов
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Сортируем по убыванию TF-IDF
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw for kw, score in keyword_scores[:50]]
    except Exception as e:
        print(f"Error extracting n-grams: {e}")
        return []


def extract_yake_keywords(documents: List[str], max_keywords: int = 50) -> List[str]:
    """Извлекает ключевые слова через YAKE"""
    if not YAKE_AVAILABLE:
        return []
    
    try:
        # Объединяем все документы
        all_text = ' '.join(documents)
        
        # Настройки YAKE
        language = "en"
        max_ngram_size = 3
        deduplication_threshold = 0.7
        numOfKeywords = max_keywords
        
        # Создаем экстрактор YAKE
        kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=deduplication_threshold,
            top=numOfKeywords,
            features=None
        )
        
        # Извлекаем ключевые слова
        keywords = kw_extractor.extract_keywords(all_text)
        
        # Возвращаем только текст ключевых слов (kw[0] - это текст, kw[1] - оценка)
        return [str(kw[0]) for kw in keywords if len(kw) > 0 and isinstance(kw[0], str)]
    except Exception as e:
        print(f"Error extracting YAKE keywords: {e}")
        return []


def extract_textrank_keywords(documents: List[str], max_keywords: int = 50) -> List[str]:
    """Извлекает ключевые слова через TextRank"""
    if not TEXTRANK_AVAILABLE:
        return []
    
    try:
        # Объединяем все документы
        all_text = ' '.join(documents)
        
        # Извлекаем ключевые слова через TextRank
        extracted_keywords = keywords.keywords(all_text, words=max_keywords)
        
        # Разбиваем по запятым и очищаем
        keyword_list = [kw.strip() for kw in extracted_keywords.split('\n') if kw.strip()]
        
        return keyword_list[:max_keywords]
    except Exception as e:
        print(f"Error extracting TextRank keywords: {e}")
        return []


def calculate_keyword_overlap(keywords1: List[str], keywords2: List[str]) -> Dict[str, float]:
    """Вычисляет метрики пересечения ключевых слов"""
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    jaccard = len(intersection) / len(union) if union else 0.0
    precision = len(intersection) / len(set1) if set1 else 0.0
    recall = len(intersection) / len(set2) if set2 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'jaccard': jaccard,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. График сравнения метрик по методам
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение методов извлечения ключевых слов', fontsize=16, fontweight='bold')
    
    methods = ['ngrams', 'yake', 'textrank']
    topics = ['text_mining', 'information_retrieval']
    
    # Подготовка данных
    jaccard_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for topic in topics:
        for method in methods:
            if topic in results and method in results[topic]:
                overlap = results[topic][method]['overlap_metrics']
                jaccard_scores.append(overlap['jaccard'])
                f1_scores.append(overlap['f1'])
                precision_scores.append(overlap['precision'])
                recall_scores.append(overlap['recall'])
    
    # График Jaccard Index
    ax1 = axes[0, 0]
    x_pos = range(len(methods) * len(topics))
    bars1 = ax1.bar(x_pos, jaccard_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'] * 2)
    ax1.set_title('Jaccard Index по методам и темам')
    ax1.set_ylabel('Jaccard Index')
    ax1.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax1.set_xticklabels(['N-grams\nTM', 'YAKE\nTM', 'TextRank\nTM', 
                        'N-grams\nIR', 'YAKE\nIR', 'TextRank\nIR'])
    ax1.grid(True, alpha=0.3)
    
    # График F1-score
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'] * 2)
    ax2.set_title('F1-score по методам и темам')
    ax2.set_ylabel('F1-score')
    ax2.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax2.set_xticklabels(['N-grams\nTM', 'YAKE\nTM', 'TextRank\nTM', 
                        'N-grams\nIR', 'YAKE\nIR', 'TextRank\nIR'])
    ax2.grid(True, alpha=0.3)
    
    # График Precision
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, precision_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'] * 2)
    ax3.set_title('Precision по методам и темам')
    ax3.set_ylabel('Precision')
    ax3.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax3.set_xticklabels(['N-grams\nTM', 'YAKE\nTM', 'TextRank\nTM', 
                        'N-grams\nIR', 'YAKE\nIR', 'TextRank\nIR'])
    ax3.grid(True, alpha=0.3)
    
    # График Recall
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, recall_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'] * 2)
    ax4.set_title('Recall по методам и темам')
    ax4.set_ylabel('Recall')
    ax4.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax4.set_xticklabels(['N-grams\nTM', 'YAKE\nTM', 'TextRank\nTM', 
                        'N-grams\nIR', 'YAKE\nIR', 'TextRank\nIR'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. График топ ключевых слов
    for topic in topics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Топ-10 ключевых слов: {topic.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        
        for i, method in enumerate(methods):
            if topic in results and method in results[topic]:
                human_kw = results[topic][method]['human_keywords'][:10]
                synthetic_kw = results[topic][method]['synthetic_keywords'][:10]
                
                # Создаем данные для графика
                all_kw = list(set(human_kw + synthetic_kw))
                human_counts = [human_kw.count(kw) for kw in all_kw]
                synthetic_counts = [synthetic_kw.count(kw) for kw in all_kw]
                
                x = np.arange(len(all_kw))
                width = 0.35
                
                ax = axes[i]
                bars1 = ax.bar(x - width/2, human_counts, width, label='Человеческие', alpha=0.8)
                bars2 = ax.bar(x + width/2, synthetic_counts, width, label='Синтетические', alpha=0.8)
                
                ax.set_title(f'{method.upper()}')
                ax.set_ylabel('Частота')
                ax.set_xticks(x)
                ax.set_xticklabels(all_kw, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_keywords_{topic}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. График разнообразия ключевых слов
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Анализ разнообразия ключевых слов', fontsize=16, fontweight='bold')
    
    # Подготовка данных для разнообразия
    human_unique_ratios = []
    synthetic_unique_ratios = []
    method_labels = []
    
    for topic in topics:
        for method in methods:
            if topic in results and method in results[topic]:
                human_div = results[topic][method]['human_diversity']
                synthetic_div = results[topic][method]['synthetic_diversity']
                human_unique_ratios.append(human_div['unique_ratio'])
                synthetic_unique_ratios.append(synthetic_div['unique_ratio'])
                method_labels.append(f'{method.upper()}\n{topic[:2].upper()}')
    
    x = np.arange(len(method_labels))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, human_unique_ratios, width, label='Человеческие', alpha=0.8)
    bars2 = ax1.bar(x + width/2, synthetic_unique_ratios, width, label='Синтетические', alpha=0.8)
    ax1.set_title('Уникальность ключевых слов')
    ax1.set_ylabel('Коэффициент уникальности')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График средней длины ключевых слов
    human_avg_lengths = []
    synthetic_avg_lengths = []
    
    for topic in topics:
        for method in methods:
            if topic in results and method in results[topic]:
                human_div = results[topic][method]['human_diversity']
                synthetic_div = results[topic][method]['synthetic_diversity']
                human_avg_lengths.append(human_div['avg_length'])
                synthetic_avg_lengths.append(synthetic_div['avg_length'])
    
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, human_avg_lengths, width, label='Человеческие', alpha=0.8)
    bars2 = ax2.bar(x + width/2, synthetic_avg_lengths, width, label='Синтетические', alpha=0.8)
    ax2.set_title('Средняя длина ключевых слов')
    ax2.set_ylabel('Количество слов')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в папке: {output_dir}")


def generate_markdown_report(results: Dict, output_path: str):
    """Генерирует отчет в формате Markdown"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Эксперимент 1: Анализ ключевых слов в человеческих и синтетических текстах\n\n")
        
        f.write("## Методология\n\n")
        f.write("- **Выборка**: 30 человеческих + 30 синтетических документов (по 15 на тему)\n")
        f.write("- **Темы**: Text Mining, Information Retrieval\n")
        f.write("- **Методы извлечения ключевых слов**:\n")
        f.write("  - N-граммы (TF-IDF, 1-3 граммы)\n")
        f.write("  - YAKE (Yet Another Keyword Extractor)\n")
        f.write("  - TextRank\n")
        f.write("- **Метрики сравнения**: Jaccard, Precision, Recall, F1-score\n\n")
        
        f.write("## Визуализация результатов\n\n")
        f.write("### Сравнение метрик по методам\n\n")
        f.write("![Сравнение метрик](metrics_comparison.png)\n\n")
        f.write("### Топ ключевых слов по темам\n\n")
        f.write("![Топ ключевых слов Text Mining](top_keywords_text_mining.png)\n\n")
        f.write("![Топ ключевых слов Information Retrieval](top_keywords_information_retrieval.png)\n\n")
        f.write("### Анализ разнообразия\n\n")
        f.write("![Анализ разнообразия](diversity_analysis.png)\n\n")
        
        f.write("## Результаты по темам\n\n")
        
        for topic in ['text_mining', 'information_retrieval']:
            f.write(f"### {topic.replace('_', ' ').title()}\n\n")
            
            topic_results = results[topic]
            
            # Статистика по документам
            f.write("#### Статистика документов\n\n")
            f.write(f"- Человеческих документов: {topic_results['human_docs_count']}\n")
            f.write(f"- Синтетических документов: {topic_results['synthetic_docs_count']}\n\n")
            
            # Результаты по методам
            for method in ['ngrams', 'yake', 'textrank']:
                f.write(f"#### {method.upper()}\n\n")
                
                method_results = topic_results[method]
                
                # Топ ключевые слова
                f.write("**Топ-10 ключевых слов (человеческие тексты):**\n")
                human_kw = method_results['human_keywords'][:10]
                for i, kw in enumerate(human_kw, 1):
                    f.write(f"{i}. {kw}\n")
                f.write("\n")
                
                f.write("**Топ-10 ключевых слов (синтетические тексты):**\n")
                synthetic_kw = method_results['synthetic_keywords'][:10]
                for i, kw in enumerate(synthetic_kw, 1):
                    f.write(f"{i}. {kw}\n")
                f.write("\n")
                
                # Метрики сравнения
                overlap = method_results['overlap_metrics']
                f.write("**Метрики пересечения:**\n")
                f.write(f"- Jaccard Index: {overlap['jaccard']:.3f}\n")
                f.write(f"- Precision: {overlap['precision']:.3f}\n")
                f.write(f"- Recall: {overlap['recall']:.3f}\n")
                f.write(f"- F1-score: {overlap['f1']:.3f}\n")
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
        
        # Общие выводы
        f.write("## Общие выводы\n\n")
        
        # Сравнение методов
        f.write("### Сравнение методов извлечения ключевых слов\n\n")
        f.write("| Метод | Средний Jaccard | Средний F1 | Средняя уникальность |\n")
        f.write("|-------|----------------|------------|---------------------|\n")
        
        method_stats = defaultdict(list)
        for topic in ['text_mining', 'information_retrieval']:
            for method in ['ngrams', 'yake', 'textrank']:
                overlap = results[topic][method]['overlap_metrics']
                human_div = results[topic][method]['human_diversity']
                method_stats[method].append({
                    'jaccard': overlap['jaccard'],
                    'f1': overlap['f1'],
                    'unique': human_div['unique_ratio']
                })
        
        for method, stats in method_stats.items():
            avg_jaccard = np.mean([s['jaccard'] for s in stats])
            avg_f1 = np.mean([s['f1'] for s in stats])
            avg_unique = np.mean([s['unique'] for s in stats])
            f.write(f"| {method.upper()} | {avg_jaccard:.3f} | {avg_f1:.3f} | {avg_unique:.3f} |\n")
        
        f.write("\n")
        
        # Выводы
        f.write("### Ключевые наблюдения\n\n")
        f.write("1. **Различия в ключевых словах**: Синтетические тексты показывают различия в выборе ключевых слов по сравнению с человеческими.\n")
        f.write("2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.\n")
        f.write("3. **Тематическая специфичность**: Каждая тема имеет свои характерные ключевые слова.\n")
        f.write("4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.\n\n")
        
        f.write("## Заключение\n\n")
        f.write("Эксперимент показал, что анализ ключевых слов может быть эффективным методом для различения человеческих и AI-сгенерированных текстов. "
               "Различные методы извлечения ключевых слов дают дополнительные возможности для анализа и могут быть объединены для повышения точности детекции.\n")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент 1: Анализ ключевых слов")
    parser.add_argument("--output_dir", default="results/experiment1", help="Папка для результатов")
    parser.add_argument("--docs_per_topic", type=int, default=15, help="Количество документов на тему")
    args = parser.parse_args()
    
    # Создаем папку для результатов
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Пути к данным
    data_paths = {
        'text_mining': {
            'human': 'data/arxiv_docs/text_mining.csv',
            'synthetic': {
                'llama': 'data/ai/llama_api_text/text_mining_full',
                'qwen': 'data/ai/qwen_api_auto/text_mining_full',
                'deepseek': 'data/ai/deepseek_api_auto/text_mining_full'
            }
        },
        'information_retrieval': {
            'human': 'data/arxiv_docs/information_retrieval.csv',
            'synthetic': {
                'llama': 'data/ai/llama_api/ir',
                'qwen': 'data/ai/qwen_api/ir',
                'deepseek': 'data/ai/deepseek_api/ir'
            }
        }
    }
    
    results = {}
    
    print("Запуск эксперимента 1: Анализ ключевых слов...")
    
    for topic, paths in data_paths.items():
        print(f"\nОбработка темы: {topic}")
        
        # Загружаем человеческие документы
        human_docs = load_documents_from_csv(paths['human'], args.docs_per_topic)
        print(f"Загружено {len(human_docs)} человеческих документов")
        
        # Загружаем синтетические документы (объединяем все модели)
        synthetic_docs = []
        for model, path in paths['synthetic'].items():
            docs = load_documents_from_txt_dir(path, args.docs_per_topic // 3)  # По 5 от каждой модели
            synthetic_docs.extend(docs)
            print(f"Загружено {len(docs)} документов от {model}")
        
        print(f"Всего синтетических документов: {len(synthetic_docs)}")
        
        if len(human_docs) == 0 or len(synthetic_docs) == 0:
            print(f"Пропускаем {topic} - недостаточно документов")
            continue
        
        topic_results = {
            'human_docs_count': len(human_docs),
            'synthetic_docs_count': len(synthetic_docs),
            'ngrams': {},
            'yake': {},
            'textrank': {}
        }
        
        # Извлекаем ключевые слова разными методами
        methods = [
            ('ngrams', extract_ngrams_keywords),
            ('yake', extract_yake_keywords),
            ('textrank', extract_textrank_keywords)
        ]
        
        for method_name, extract_func in methods:
            print(f"  Извлечение ключевых слов методом {method_name}...")
            
            # Ключевые слова для человеческих текстов
            human_keywords = extract_func(human_docs)
            
            # Ключевые слова для синтетических текстов
            synthetic_keywords = extract_func(synthetic_docs)
            
            # Вычисляем метрики пересечения
            overlap_metrics = calculate_keyword_overlap(human_keywords, synthetic_keywords)
            
            # Анализируем разнообразие
            human_diversity = analyze_keyword_diversity(human_keywords)
            synthetic_diversity = analyze_keyword_diversity(synthetic_keywords)
            
            topic_results[method_name] = {
                'human_keywords': human_keywords,
                'synthetic_keywords': synthetic_keywords,
                'overlap_metrics': overlap_metrics,
                'human_diversity': human_diversity,
                'synthetic_diversity': synthetic_diversity
            }
            
            print(f"    Найдено ключевых слов: человеческие {len(human_keywords)}, синтетические {len(synthetic_keywords)}")
            print(f"    Jaccard: {overlap_metrics['jaccard']:.3f}, F1: {overlap_metrics['f1']:.3f}")
        
        results[topic] = topic_results
    
    # Сохраняем результаты в JSON
    json_path = os.path.join(args.output_dir, 'experiment1_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Создаем графики
    create_visualizations(results, args.output_dir)
    
    # Генерируем отчет
    report_path = os.path.join(args.output_dir, 'experiment1_report.md')
    generate_markdown_report(results, report_path)
    
    print(f"\nЭксперимент завершен!")
    print(f"Результаты сохранены в: {args.output_dir}")
    print(f"Отчет: {report_path}")
    print(f"JSON данные: {json_path}")


if __name__ == "__main__":
    main()
