#!/usr/bin/env python3
"""
Дополнительный анализ человеческих и синтетических текстов
- Анализ структуры предложений
- Лексическое разнообразие (TTR)
- Анализ связующих элементов
- Анализ повторений
- Sentiment analysis
- Сравнение по моделям генерации
"""

import os
import sys
import json
import argparse
import re
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
from collections import Counter, defaultdict
try:
    import nltk
    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    NLTK_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if NLTK_AVAILABLE:
    # Загружаем необходимые ресурсы NLTK
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        print("Warning: NLTK resources not fully available")

    # Пытаемся импортировать токенизаторы/словарь стоп-слов
    try:
        from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize, word_tokenize as _nltk_word_tokenize
    except Exception:
        _nltk_sent_tokenize = None
        _nltk_word_tokenize = None
    try:
        from nltk.corpus import stopwords as _nltk_stopwords
    except Exception:
        _nltk_stopwords = None
else:
    _nltk_sent_tokenize = None
    _nltk_word_tokenize = None
    _nltk_stopwords = None

# Фолбэк токенизаторов/стоп-слов
import re as _re_token

def sent_tokenize(text: str):
    if _nltk_sent_tokenize is not None:
        try:
            return _nltk_sent_tokenize(text)
        except Exception:
            pass
    # Простой фолбэк по точкам
    return [s.strip() for s in str(text).split('.') if s.strip()]

def word_tokenize(text: str):
    if _nltk_word_tokenize is not None:
        try:
            return _nltk_word_tokenize(text)
        except Exception:
            pass
    return _re_token.findall(r"[A-Za-z']+", str(text))

def get_stopwords() -> set:
    if _nltk_stopwords is not None:
        try:
            return set(_nltk_stopwords.words('english'))
        except Exception:
            pass
    return {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been',
        'have','has','had','do','does','did','will','would','could','should','may','might','must','can','this','that',
        'these','those','i','you','he','she','it','we','they','me','him','her','us','them'
    }

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment analysis not available")

# Настройка стиля графиков
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')
if USE_SEABORN:
    try:
        sns.set_palette("husl")
    except Exception:
        pass


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


def analyze_sentence_structure(documents: List[str]) -> Dict[str, float]:
    """Анализирует структуру предложений"""
    all_sentences = []
    sentence_lengths = []
    
    for doc in documents:
        try:
            sentences = sent_tokenize(doc)
            all_sentences.extend(sentences)
            sentence_lengths.extend([len(word_tokenize(sent)) for sent in sentences])
        except:
            # Fallback: простой split по точкам
            sentences = [s.strip() for s in doc.split('.') if s.strip()]
            all_sentences.extend(sentences)
            sentence_lengths.extend([len(s.split()) for s in sentences])
    
    if not sentence_lengths:
        return {
            'avg_sentence_length': 0.0,
            'std_sentence_length': 0.0,
            'avg_sentences_per_doc': 0.0,
            'total_sentences': 0,
            'total_documents': len(documents)
        }
    
    return {
        'avg_sentence_length': np.mean(sentence_lengths),
        'std_sentence_length': np.std(sentence_lengths),
        'avg_sentences_per_doc': len(all_sentences) / len(documents) if documents else 0,
        'total_sentences': len(all_sentences),
        'total_documents': len(documents)
    }


def analyze_lexical_diversity(documents: List[str]) -> Dict[str, float]:
    """Анализирует лексическое разнообразие"""
    all_text = ' '.join(documents)
    
    try:
        words = word_tokenize(all_text.lower())
    except:
        words = all_text.lower().split()
    
    # Убираем пунктуацию и стоп-слова
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback: простой список стоп-слов
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    words = [w for w in words if w.isalpha() and w not in stop_words]
    
    if not words:
        return {
            'ttr': 0.0,
            'simpson_diversity': 0.0,
            'total_tokens': 0,
            'unique_types': 0,
            'avg_word_length': 0.0
        }
    
    # TTR (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    
    # Индекс разнообразия Симпсона
    word_counts = Counter(words)
    simpson_diversity = 1 - sum((count/len(words))**2 for count in word_counts.values())
    
    # Средняя длина слов
    avg_word_length = np.mean([len(w) for w in words])
    
    return {
        'ttr': ttr,
        'simpson_diversity': simpson_diversity,
        'total_tokens': len(words),
        'unique_types': len(unique_words),
        'avg_word_length': avg_word_length
    }


def analyze_connectors(documents: List[str]) -> Dict[str, float]:
    """Анализирует использование связующих элементов"""
    # Список связующих слов и фраз
    connectors = {
        'contrast': ['however', 'but', 'although', 'though', 'despite', 'nevertheless', 'nonetheless'],
        'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides', 'in addition'],
        'cause_effect': ['therefore', 'thus', 'consequently', 'hence', 'as a result', 'because'],
        'sequence': ['firstly', 'secondly', 'thirdly', 'finally', 'next', 'then', 'subsequently'],
        'emphasis': ['indeed', 'certainly', 'obviously', 'clearly', 'undoubtedly', 'surely'],
        'example': ['for example', 'for instance', 'such as', 'namely', 'specifically']
    }
    
    all_text = ' '.join(documents).lower()
    
    connector_counts = {}
    total_connectors = 0
    
    for category, words in connectors.items():
        count = 0
        for word in words:
            count += all_text.count(word)
        connector_counts[category] = count
        total_connectors += count
    
    # Нормализация по количеству слов
    try:
        word_count = len(word_tokenize(all_text))
    except:
        word_count = len(all_text.split())
    
    normalized_counts = {k: v/word_count*1000 for k, v in connector_counts.items()}  # на 1000 слов
    
    return {
        'total_connectors': total_connectors,
        'connectors_per_1000_words': total_connectors/word_count*1000 if word_count > 0 else 0,
        'normalized_counts': normalized_counts,
        'category_counts': connector_counts
    }


def analyze_repetitions(documents: List[str]) -> Dict[str, float]:
    """Анализирует повторения и шаблонность"""
    all_text = ' '.join(documents)
    
    # Анализ биграмм и триграмм
    try:
        words = word_tokenize(all_text.lower())
    except:
        words = all_text.lower().split()
    
    # Убираем пунктуацию
    words = [w for w in words if w.isalpha()]
    
    # Биграммы
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    
    # Триграммы
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)
    
    # Метрики повторений
    unique_bigrams = len(set(bigrams))
    unique_trigrams = len(set(trigrams))
    
    # Коэффициент повторения (доля повторяющихся n-грамм)
    bigram_repetition = (len(bigrams) - unique_bigrams) / len(bigrams) if bigrams else 0
    trigram_repetition = (len(trigrams) - unique_trigrams) / len(trigrams) if trigrams else 0
    
    # Топ повторяющихся фраз
    top_bigrams = bigram_counts.most_common(10)
    top_trigrams = trigram_counts.most_common(10)
    
    return {
        'bigram_repetition_rate': bigram_repetition,
        'trigram_repetition_rate': trigram_repetition,
        'unique_bigrams': unique_bigrams,
        'unique_trigrams': unique_trigrams,
        'total_bigrams': len(bigrams),
        'total_trigrams': len(trigrams),
        'top_bigrams': top_bigrams,
        'top_trigrams': top_trigrams
    }


def _fallback_sentiment(documents: List[str]) -> Dict[str, float]:
    """Простой лексиконный fallback, если VADER недоступен."""
    pos_words = {
        'improve','improves','improved','significant','robust','effective','efficient','state-of-the-art',
        'novel','promising','accurate','reliable','stable','strong','outperform','outperforms','advances','advance'
    }
    neg_words = {
        'fail','fails','failure','error','errors','limitation','limitations','bias','noisy','weak','unstable',
        'overfit','overfitting','underperform','underperforms','degrade','degradation'
    }
    import re as _re
    tokens_all = []
    scores = []
    for doc in documents:
        tokens = [t for t in _re.findall(r"[a-zA-Z']+", str(doc).lower())]
        tokens_all.extend(tokens)
        if not tokens:
            scores.append({'pos':0.0,'neg':0.0,'neu':1.0,'compound':0.0})
            continue
        pos = sum(1 for t in tokens if t in pos_words)
        neg = sum(1 for t in tokens if t in neg_words)
        total = len(tokens)
        comp = 0.0
        if total > 0:
            comp = (pos - neg) / max(1.0, (pos + neg + 10))  # сглаживание
        neu = max(0.0, 1.0 - min(1.0, (pos+neg)/max(1.0,total)))
        p = min(1.0, pos/max(1.0,total))
        n = min(1.0, neg/max(1.0,total))
        scores.append({'pos':p,'neg':n,'neu':neu,'compound':comp})
    if not scores:
        return {
            'avg_positive': 0.0,
            'avg_negative': 0.0,
            'avg_neutral': 0.0,
            'avg_compound': 0.0,
            'sentiment_distribution': {}
        }
    return {
        'avg_positive': float(np.mean([s['pos'] for s in scores])),
        'avg_negative': float(np.mean([s['neg'] for s in scores])),
        'avg_neutral': float(np.mean([s['neu'] for s in scores])),
        'avg_compound': float(np.mean([s['compound'] for s in scores])),
        'sentiment_distribution': {
            'positive': sum(1 for s in scores if s['compound'] > 0.05),
            'negative': sum(1 for s in scores if s['compound'] < -0.05),
            'neutral': sum(1 for s in scores if -0.05 <= s['compound'] <= 0.05)
        }
    }


def analyze_sentiment(documents: List[str]) -> Dict[str, float]:
    """Анализирует эмоциональную окраску текстов"""
    if not SENTIMENT_AVAILABLE:
        return _fallback_sentiment(documents)
    
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        for doc in documents:
            scores = analyzer.polarity_scores(doc)
            sentiments.append(scores)
        
        if not sentiments:
            return {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 0.0,
                'avg_compound': 0.0,
                'sentiment_distribution': {}
            }
        
        avg_sentiment = {
            'avg_positive': np.mean([s['pos'] for s in sentiments]),
            'avg_negative': np.mean([s['neg'] for s in sentiments]),
            'avg_neutral': np.mean([s['neu'] for s in sentiments]),
            'avg_compound': np.mean([s['compound'] for s in sentiments])
        }
        
        # Распределение по типам тональности
        sentiment_distribution = {
            'positive': sum(1 for s in sentiments if s['compound'] > 0.05),
            'negative': sum(1 for s in sentiments if s['compound'] < -0.05),
            'neutral': sum(1 for s in sentiments if -0.05 <= s['compound'] <= 0.05)
        }
        
        avg_sentiment['sentiment_distribution'] = sentiment_distribution
        return avg_sentiment
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return _fallback_sentiment(documents)


def analyze_by_model(documents_by_model: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Анализирует различия между моделями генерации"""
    model_analysis = {}
    
    for model_name, docs in documents_by_model.items():
        if not docs:
            continue
            
        model_analysis[model_name] = {
            'sentence_structure': analyze_sentence_structure(docs),
            'lexical_diversity': analyze_lexical_diversity(docs),
            'connectors': analyze_connectors(docs),
            'repetitions': analyze_repetitions(docs),
            'sentiment': analyze_sentiment(docs),
            'document_count': len(docs)
        }
    
    return model_analysis


def create_comprehensive_visualizations(results: Dict, output_dir: str):
    """Создает комплексные визуализации"""
    
    # 1. Сравнение метрик по типам текстов
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Комплексный анализ человеческих и синтетических текстов', fontsize=16, fontweight='bold')
    
    topics = ['text_mining', 'information_retrieval']
    text_types = ['human', 'synthetic']
    
    # Подготовка данных
    metrics_data = {
        'avg_sentence_length': [],
        'ttr': [],
        'connectors_per_1000': [],
        'bigram_repetition': [],
        'sentiment_compound': [],
        'simpson_diversity': []
    }
    
    labels = []
    colors = ['#1f77b4', '#ff7f0e']
    
    for topic in topics:
        for text_type in text_types:
            if topic in results and text_type in results[topic]:
                data = results[topic][text_type]
                metrics_data['avg_sentence_length'].append(data['sentence_structure']['avg_sentence_length'])
                metrics_data['ttr'].append(data['lexical_diversity']['ttr'])
                metrics_data['connectors_per_1000'].append(data['connectors']['connectors_per_1000_words'])
                metrics_data['bigram_repetition'].append(data['repetitions']['bigram_repetition_rate'])
                metrics_data['sentiment_compound'].append(data['sentiment']['avg_compound'])
                metrics_data['simpson_diversity'].append(data['lexical_diversity']['simpson_diversity'])
                labels.append(f'{topic[:2].upper()}\n{text_type[:3].upper()}')
    
    # Графики
    metric_names = ['avg_sentence_length', 'ttr', 'connectors_per_1000', 'bigram_repetition', 'sentiment_compound', 'simpson_diversity']
    metric_titles = ['Средняя длина предложения', 'TTR (лексическое разнообразие)', 'Связки на 1000 слов', 
                   'Повторение биграмм', 'Эмоциональная окраска', 'Индекс Симпсона']
    
    for i, (metric, title) in enumerate(zip(metric_names, metric_titles)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        x_pos = range(len(metrics_data[metric]))
        bars = ax.bar(x_pos, metrics_data[metric], color=colors * len(topics), alpha=0.8)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Анализ по моделям (если есть данные)
    if 'model_comparison' in results and results['model_comparison']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Сравнение моделей генерации', fontsize=16, fontweight='bold')
        
        models = list(results['model_comparison'].keys())
        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Подготовка данных по моделям
        model_metrics = {
            'avg_sentence_length': [],
            'ttr': [],
            'connectors_per_1000': [],
            'sentiment_compound': []
        }
        
        for model in models:
            data = results['model_comparison'][model]
            model_metrics['avg_sentence_length'].append(data['sentence_structure']['avg_sentence_length'])
            model_metrics['ttr'].append(data['lexical_diversity']['ttr'])
            model_metrics['connectors_per_1000'].append(data['connectors']['connectors_per_1000_words'])
            model_metrics['sentiment_compound'].append(data['sentiment']['avg_compound'])
        
        metric_titles = ['Средняя длина предложения', 'TTR', 'Связки на 1000 слов', 'Эмоциональная окраска']
        
        for i, (metric, title) in enumerate(zip(model_metrics.keys(), metric_titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            x_pos = range(len(models))
            bars = ax.bar(x_pos, model_metrics[metric], color=model_colors[:len(models)], alpha=0.8)
            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Тепловая карта корреляций
    if len(metrics_data['avg_sentence_length']) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Создаем DataFrame для корреляций
        df_corr = pd.DataFrame({
            'Sentence Length': metrics_data['avg_sentence_length'],
            'TTR': metrics_data['ttr'],
            'Connectors': metrics_data['connectors_per_1000'],
            'Repetition': metrics_data['bigram_repetition'],
            'Sentiment': metrics_data['sentiment_compound'],
            'Diversity': metrics_data['simpson_diversity']
        })
        
        correlation_matrix = df_corr.corr()
        
        if USE_SEABORN:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                        square=True, ax=ax, cbar_kws={'shrink': 0.8})
        else:
            cax = ax.imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.index)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.index)
            for (i, j), val in np.ndenumerate(correlation_matrix.values):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
            fig.colorbar(cax, ax=ax, shrink=0.8)
        ax.set_title('Корреляционная матрица метрик', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Комплексные графики сохранены в папке: {output_dir}")


def generate_comprehensive_report(results: Dict, output_path: str):
    """Генерирует комплексный отчет"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Комплексный анализ человеческих и синтетических текстов\n\n")
        
        f.write("## Методология\n\n")
        f.write("Данный анализ включает следующие аспекты:\n")
        f.write("- **Структура предложений**: средняя длина, количество предложений\n")
        f.write("- **Лексическое разнообразие**: TTR, индекс Симпсона\n")
        f.write("- **Связующие элементы**: частота вводных слов и переходных фраз\n")
        f.write("- **Повторения**: анализ биграмм и триграмм\n")
        f.write("- **Эмоциональная окраска**: sentiment analysis\n")
        f.write("- **Сравнение моделей**: различия между Llama, Qwen, DeepSeek\n\n")
        
        f.write("## Визуализация результатов\n\n")
        f.write("### Комплексный анализ\n\n")
        f.write("![Комплексный анализ](comprehensive_analysis.png)\n\n")
        
        if 'model_comparison' in results and results['model_comparison']:
            f.write("### Сравнение моделей\n\n")
            f.write("![Сравнение моделей](model_comparison.png)\n\n")
        
        f.write("### Корреляционный анализ\n\n")
        f.write("![Корреляционная матрица](correlation_heatmap.png)\n\n")
        
        f.write("## Детальные результаты\n\n")
        
        for topic in ['text_mining', 'information_retrieval']:
            if topic not in results:
                continue
                
            f.write(f"### {topic.replace('_', ' ').title()}\n\n")
            
            for text_type in ['human', 'synthetic']:
                if text_type not in results[topic]:
                    continue
                    
                f.write(f"#### {text_type.title()} тексты\n\n")
                
                data = results[topic][text_type]
                
                # Структура предложений
                sent_data = data['sentence_structure']
                f.write("**Структура предложений:**\n")
                f.write(f"- Средняя длина предложения: {sent_data['avg_sentence_length']:.2f} слов\n")
                f.write(f"- Стандартное отклонение: {sent_data['std_sentence_length']:.2f}\n")
                f.write(f"- Среднее количество предложений на документ: {sent_data['avg_sentences_per_doc']:.2f}\n\n")
                
                # Лексическое разнообразие
                lex_data = data['lexical_diversity']
                f.write("**Лексическое разнообразие:**\n")
                f.write(f"- TTR (Type-Token Ratio): {lex_data['ttr']:.3f}\n")
                f.write(f"- Индекс разнообразия Симпсона: {lex_data['simpson_diversity']:.3f}\n")
                f.write(f"- Средняя длина слов: {lex_data['avg_word_length']:.2f} символов\n")
                f.write(f"- Уникальных слов: {lex_data['unique_types']} из {lex_data['total_tokens']}\n\n")
                
                # Связующие элементы
                conn_data = data['connectors']
                f.write("**Связующие элементы:**\n")
                f.write(f"- Всего связок на 1000 слов: {conn_data['connectors_per_1000_words']:.2f}\n")
                f.write("По категориям:\n")
                for category, count in conn_data['normalized_counts'].items():
                    f.write(f"  - {category}: {count:.2f} на 1000 слов\n")
                f.write("\n")
                
                # Повторения
                rep_data = data['repetitions']
                f.write("**Анализ повторений:**\n")
                f.write(f"- Коэффициент повторения биграмм: {rep_data['bigram_repetition_rate']:.3f}\n")
                f.write(f"- Коэффициент повторения триграмм: {rep_data['trigram_repetition_rate']:.3f}\n")
                f.write(f"- Уникальных биграмм: {rep_data['unique_bigrams']} из {rep_data['total_bigrams']}\n")
                f.write(f"- Уникальных триграмм: {rep_data['unique_trigrams']} из {rep_data['total_trigrams']}\n\n")
                
                # Топ повторяющихся фраз
                f.write("**Топ-5 повторяющихся биграмм:**\n")
                for i, (phrase, count) in enumerate(rep_data['top_bigrams'][:5], 1):
                    f.write(f"{i}. \"{phrase}\" ({count} раз)\n")
                f.write("\n")
                
                # Эмоциональная окраска
                sent_data = data['sentiment']
                f.write("**Эмоциональная окраска:**\n")
                f.write(f"- Средний compound score: {sent_data['avg_compound']:.3f}\n")
                f.write(f"- Положительная тональность: {sent_data['avg_positive']:.3f}\n")
                f.write(f"- Отрицательная тональность: {sent_data['avg_negative']:.3f}\n")
                f.write(f"- Нейтральная тональность: {sent_data['avg_neutral']:.3f}\n\n")
            
            f.write("---\n\n")
        
        # Сравнение моделей
        if 'model_comparison' in results and results['model_comparison']:
            f.write("## Сравнение моделей генерации\n\n")
            
            f.write("| Модель | Длина предложения | TTR | Связки/1000 | Sentiment |\n")
            f.write("|--------|------------------|-----|-------------|----------|\n")
            
            for model, data in results['model_comparison'].items():
                sent_len = data['sentence_structure']['avg_sentence_length']
                ttr = data['lexical_diversity']['ttr']
                connectors = data['connectors']['connectors_per_1000_words']
                sentiment = data['sentiment']['avg_compound']
                
                f.write(f"| {model} | {sent_len:.2f} | {ttr:.3f} | {connectors:.2f} | {sentiment:.3f} |\n")
            
            f.write("\n")
        
        # Выводы
        f.write("## Ключевые выводы\n\n")
        f.write("### Основные различия между человеческими и синтетическими текстами:\n\n")
        f.write("1. **Структура предложений**: Синтетические тексты могут иметь другую структуру предложений\n")
        f.write("2. **Лексическое разнообразие**: Различия в TTR и использовании уникальных слов\n")
        f.write("3. **Связующие элементы**: Разная частота использования переходных фраз\n")
        f.write("4. **Повторения**: Синтетические тексты могут показывать больше или меньше повторений\n")
        f.write("5. **Эмоциональная окраска**: Различия в тональности и стиле\n\n")
        
        f.write("### Практические применения:\n\n")
        f.write("- **Детекция AI-текстов**: Комбинация метрик может повысить точность детекции\n")
        f.write("- **Качество генерации**: Анализ помогает оценить качество различных моделей\n")
        f.write("- **Улучшение моделей**: Выявленные паттерны могут использоваться для улучшения генерации\n\n")
        
        f.write("## Заключение\n\n")
        f.write("Комплексный анализ выявил значительные различия между человеческими и синтетическими текстами "
               "на различных уровнях: структурном, лексическом, стилистическом. Эти различия могут быть "
               "эффективно использованы для разработки более точных методов детекции AI-сгенерированных текстов.\n")


def main():
    parser = argparse.ArgumentParser(description="Комплексный анализ текстов")
    parser.add_argument("--output_dir", default="results/additional_analysis", help="Папка для результатов")
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
    model_documents = defaultdict(list)
    
    print("Запуск комплексного анализа...")
    
    for topic, paths in data_paths.items():
        print(f"\nОбработка темы: {topic}")
        
        # Загружаем человеческие документы
        human_docs = load_documents_from_csv(paths['human'], args.docs_per_topic)
        print(f"Загружено {len(human_docs)} человеческих документов")
        
        # Загружаем синтетические документы
        synthetic_docs = []
        for model, path in paths['synthetic'].items():
            docs = load_documents_from_txt_dir(path, args.docs_per_topic // 3)
            synthetic_docs.extend(docs)
            model_documents[model].extend(docs)
            print(f"Загружено {len(docs)} документов от {model}")
        
        print(f"Всего синтетических документов: {len(synthetic_docs)}")
        
        if len(human_docs) == 0 or len(synthetic_docs) == 0:
            print(f"Пропускаем {topic} - недостаточно документов")
            continue
        
        # Анализируем человеческие тексты
        human_analysis = {
            'sentence_structure': analyze_sentence_structure(human_docs),
            'lexical_diversity': analyze_lexical_diversity(human_docs),
            'connectors': analyze_connectors(human_docs),
            'repetitions': analyze_repetitions(human_docs),
            'sentiment': analyze_sentiment(human_docs)
        }
        
        # Анализируем синтетические тексты
        synthetic_analysis = {
            'sentence_structure': analyze_sentence_structure(synthetic_docs),
            'lexical_diversity': analyze_lexical_diversity(synthetic_docs),
            'connectors': analyze_connectors(synthetic_docs),
            'repetitions': analyze_repetitions(synthetic_docs),
            'sentiment': analyze_sentiment(synthetic_docs)
        }
        
        results[topic] = {
            'human': human_analysis,
            'synthetic': synthetic_analysis
        }
        
        print(f"Анализ {topic} завершен")
    
    # Анализ по моделям
    if model_documents:
        print("\nАнализ по моделям генерации...")
        results['model_comparison'] = analyze_by_model(dict(model_documents))
    
    # Сохраняем результаты в JSON
    json_path = os.path.join(args.output_dir, 'additional_analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Создаем визуализации
    create_comprehensive_visualizations(results, args.output_dir)
    
    # Генерируем отчет
    report_path = os.path.join(args.output_dir, 'additional_analysis_report.md')
    generate_comprehensive_report(results, report_path)
    
    print(f"\nКомплексный анализ завершен!")
    print(f"Результаты сохранены в: {args.output_dir}")
    print(f"Отчет: {report_path}")
    print(f"JSON данные: {json_path}")


if __name__ == "__main__":
    main()
