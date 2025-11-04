#!/usr/bin/env python3
"""
Эксперимент 4: Классификация HUMAN vs AI на основе ключевых слов
- Извлечение ключевых слов методами: TF-IDF n-grams, YAKE, TextRank
- Создание векторов признаков (присутствие ключевых слов в документе)
- Обучение классификаторов (Logistic Regression, SVM, Random Forest)
- Тестирование на разных количествах ключевых слов: 5, 10, 25, 40, 50
- Метрики: Accuracy, Precision, Recall, F1-score, ROC AUC
- Отчет: results/experiment4/experiment4_report.md
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Реиспользуем функции из эксперимента 1
sys.path.append(os.path.dirname(__file__))
import run_experiment1_keywords as exp1

try:
    import yake
    YAKE_AVAILABLE = True
except Exception:
    YAKE_AVAILABLE = False

try:
    from summa import keywords as summa_keywords
    TEXTRANK_AVAILABLE = True
except Exception:
    TEXTRANK_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


# ------------------------
# Извлечение ключевых слов из корпуса
# ------------------------

def extract_corpus_keywords_ngrams(documents: List[str], top_k: int = 50, 
                                   ngram_range=(1, 2), max_features=8000) -> List[str]:
    """Извлекает топ-K ключевых слов из всего корпуса методом TF-IDF n-grams"""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        preprocessor=exp1.preprocess_text,
        min_df=1,
        max_df=0.9
    )
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [feature_names[i] for i in top_indices]


def extract_corpus_keywords_yake(documents: List[str], top_k: int = 50) -> List[str]:
    """Извлекает топ-K ключевых слов из всего корпуса методом YAKE"""
    if not YAKE_AVAILABLE:
        return []
    
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.8, top=100, features=None)
    all_keywords = Counter()
    
    for doc in documents:
        try:
            kws = kw_extractor.extract_keywords(doc)
            for kw, _score in kws:
                if isinstance(kw, str):
                    all_keywords[kw] += 1
        except Exception:
            continue
    
    return [kw for kw, _ in all_keywords.most_common(top_k)]


def extract_corpus_keywords_textrank(documents: List[str], top_k: int = 50) -> List[str]:
    """Извлекает топ-K ключевых слов из всего корпуса методом TextRank"""
    if not TEXTRANK_AVAILABLE:
        return []
    
    all_keywords = Counter()
    for doc in documents:
        try:
            doc_kws = summa_keywords.keywords(doc, ratio=0.2, split=True)
            for kw in doc_kws:
                all_keywords[kw] += 1
        except Exception:
            continue
    
    return [kw for kw, _ in all_keywords.most_common(top_k)]


# ------------------------
# Создание векторов признаков
# ------------------------

def create_keyword_features(documents: List[str], keyword_list: List[str]) -> np.ndarray:
    """
    Создает матрицу признаков: для каждого документа вектор присутствия ключевых слов.
    
    Args:
        documents: Список документов
        keyword_list: Список ключевых слов
    
    Returns:
        Матрица размерности (len(documents), len(keyword_list)), где
        элемент [i, j] = 1, если keyword_list[j] присутствует в documents[i], иначе 0
    """
    features = np.zeros((len(documents), len(keyword_list)), dtype=np.float32)
    
    for i, doc in enumerate(documents):
        clean_doc = exp1.preprocess_text(doc)
        doc_lower = clean_doc.lower()
        
        for j, keyword in enumerate(keyword_list):
            # Проверяем наличие ключевого слова в документе
            keyword_clean = keyword.lower().strip()
            if keyword_clean in doc_lower:
                features[i, j] = 1.0
    
    return features


# ------------------------
# Классификация
# ------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name: str, 
                       classifier) -> Dict:
    """Обучает классификатор и вычисляет метрики"""
    
    # Обучение
    classifier.fit(X_train, y_train)
    
    # Предсказания
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
    
    # Метрики
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='binary'))
    recall = float(recall_score(y_test, y_pred, average='binary'))
    f1 = float(f1_score(y_test, y_pred, average='binary'))
    
    # Cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    
    # ROC AUC
    auc = None
    fpr, tpr = None, None
    if y_pred_proba is not None:
        try:
            auc = float(roc_auc_score(y_test, y_pred_proba))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        except Exception:
            auc = float('nan')
    
    return {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if auc is not None else float('nan'),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fpr': fpr.tolist() if fpr is not None else None,
        'tpr': tpr.tolist() if tpr is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def run_classification_experiment(human_docs: List[str], ai_docs: List[str],
                                  extraction_method: str, top_k: int) -> Dict:
    """Запускает эксперимент классификации для одного метода и одного K"""
    
    print(f"  Метод: {extraction_method}, K={top_k}...")
    
    # Извлечение ключевых слов из объединенного корпуса
    all_docs = human_docs + ai_docs
    if extraction_method == 'ngrams':
        keywords = extract_corpus_keywords_ngrams(all_docs, top_k=top_k)
    elif extraction_method == 'yake':
        keywords = extract_corpus_keywords_yake(all_docs, top_k=top_k)
    elif extraction_method == 'textrank':
        keywords = extract_corpus_keywords_textrank(all_docs, top_k=top_k)
    else:
        return {}
    
    if len(keywords) == 0:
        print(f"    Предупреждение: не удалось извлечь ключевые слова для {extraction_method}")
        return {}
    
    print(f"    Извлечено {len(keywords)} уникальных ключевых слов")
    
    # Создание признаков
    human_features = create_keyword_features(human_docs, keywords)
    ai_features = create_keyword_features(ai_docs, keywords)
    
    # Объединение и метки
    X = np.vstack([human_features, ai_features])
    y = np.array([0] * len(human_docs) + [1] * len(ai_docs))
    
    # Разделение на train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Обучение классификаторов
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    results['LogisticRegression'] = train_and_evaluate(
        X_train, X_test, y_train, y_test, 'LogisticRegression', lr
    )
    
    # Linear SVM
    svm = SVC(kernel='linear', random_state=42, probability=True)
    results['LinearSVM'] = train_and_evaluate(
        X_train, X_test, y_train, y_test, 'LinearSVM', svm
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    results['RandomForest'] = train_and_evaluate(
        X_train, X_test, y_train, y_test, 'RandomForest', rf
    )
    
    return {
        'method': extraction_method,
        'top_k': top_k,
        'keywords': keywords,
        'results': results
    }


# ------------------------
# Визуализация
# ------------------------

def create_visualizations(all_results: Dict, output_dir: str):
    """Создает графики с результатами"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Подготовка данных для графиков
    methods = ['ngrams', 'yake', 'textrank']
    k_values = [5, 10, 25, 40, 50]
    classifiers = ['LogisticRegression', 'LinearSVM', 'RandomForest']
    
    # 1. График Accuracy по методам и K
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        for method in methods:
            accuracies = []
            for k in k_values:
                key = f"{method}_{k}"
                if key in all_results and classifier in all_results[key].get('results', {}):
                    acc = all_results[key]['results'][classifier].get('accuracy', 0)
                    accuracies.append(acc)
                else:
                    accuracies.append(0)
            ax.plot(k_values, accuracies, marker='o', label=method.upper(), linewidth=2)
        
        ax.set_xlabel('Количество ключевых слов (K)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. График F1-score по методам и K
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        for method in methods:
            f1_scores = []
            for k in k_values:
                key = f"{method}_{k}"
                if key in all_results and classifier in all_results[key].get('results', {}):
                    f1 = all_results[key]['results'][classifier].get('f1', 0)
                    f1_scores.append(f1)
                else:
                    f1_scores.append(0)
            ax.plot(k_values, f1_scores, marker='s', label=method.upper(), linewidth=2)
        
        ax.set_xlabel('Количество ключевых слов (K)', fontsize=11)
        ax.set_ylabel('F1-score', fontsize=11)
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. График ROC AUC по методам и K
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        for method in methods:
            aucs = []
            for k in k_values:
                key = f"{method}_{k}"
                if key in all_results and classifier in all_results[key].get('results', {}):
                    auc = all_results[key]['results'][classifier].get('auc', 0)
                    if not np.isnan(auc):
                        aucs.append(auc)
                    else:
                        aucs.append(0)
                else:
                    aucs.append(0)
            ax.plot(k_values, aucs, marker='^', label=method.upper(), linewidth=2)
        
        ax.set_xlabel('Количество ключевых слов (K)', fontsize=11)
        ax.set_ylabel('ROC AUC', fontsize=11)
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_by_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Сводная таблица лучших результатов
    best_results = []
    for method in methods:
        for k in k_values:
            key = f"{method}_{k}"
            if key in all_results:
                for classifier in classifiers:
                    if classifier in all_results[key].get('results', {}):
                        res = all_results[key]['results'][classifier]
                        best_results.append({
                            'method': method.upper(),
                            'K': k,
                            'classifier': classifier,
                            'accuracy': res.get('accuracy', 0),
                            'f1': res.get('f1', 0),
                            'auc': res.get('auc', 0),
                        })
    
    if best_results:
        df = pd.DataFrame(best_results)
        
        # Heatmap для Accuracy
        pivot_acc = df.pivot_table(values='accuracy', index=['method'], 
                                   columns=['K', 'classifier'], aggfunc='mean')
        plt.figure(figsize=(15, 5))
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        plt.title('Accuracy по методам, K и классификаторам', fontsize=14, fontweight='bold')
        plt.xlabel('K и Классификатор', fontsize=11)
        plt.ylabel('Метод извлечения', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ------------------------
# Отчет
# ------------------------

def write_report(all_results: Dict, output_dir: str):
    """Генерирует Markdown отчет"""
    report_path = os.path.join(output_dir, 'experiment4_report.md')
    
    methods = ['ngrams', 'yake', 'textrank']
    k_values = [5, 10, 25, 40, 50]
    classifiers = ['LogisticRegression', 'LinearSVM', 'RandomForest']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Эксперимент 4: Классификация HUMAN vs AI на основе ключевых слов\n\n')
        
        f.write('## Методология\n\n')
        f.write('- **Цель**: Создание классификатора для детекции синтетических текстов на основе присутствия ключевых слов\n')
        f.write('- **Методы извлечения ключевых слов**: TF-IDF n-grams, YAKE, TextRank\n')
        f.write('- **Классификаторы**: Logistic Regression, Linear SVM, Random Forest\n')
        f.write('- **Количество ключевых слов**: K ∈ {5, 10, 25, 40, 50}\n')
        f.write('- **Векторы признаков**: Бинарные (присутствие/отсутствие ключевого слова в документе)\n')
        f.write('- **Разделение данных**: Train/Test = 80/20 с стратификацией\n')
        f.write('- **Метрики**: Accuracy, Precision, Recall, F1-score, ROC AUC, 5-fold CV\n\n')
        
        f.write('## Формулы и метрики\n\n')
        f.write('### Создание признаков\n\n')
        f.write('Для каждого документа создается бинарный вектор размерности K:\n')
        f.write('- x[i] = 1, если i-е ключевое слово присутствует в документе\n')
        f.write('- x[i] = 0, если i-е ключевое слово отсутствует в документе\n\n')
        
        f.write('### Метрики классификации\n\n')
        f.write('- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`\n')
        f.write('- **Precision**: `TP / (TP + FP)`\n')
        f.write('- **Recall**: `TP / (TP + FN)`\n')
        f.write('- **F1-score**: `2 × (Precision × Recall) / (Precision + Recall)`\n')
        f.write('- **ROC AUC**: Площадь под ROC-кривой (1.0 = идеальная разделимость)\n')
        f.write('- **CV Mean/Std**: Среднее и стандартное отклонение Accuracy на 5-fold cross-validation\n\n')
        
        f.write('## Результаты\n\n')
        
        # Результаты по каждому методу и K
        for method in methods:
            f.write(f'### Метод: {method.upper()}\n\n')
            
            # Сводная таблица
            f.write('| K | Классификатор | Accuracy | Precision | Recall | F1 | AUC | CV Mean | CV Std |\n')
            f.write('|---|---------------|----------|-----------|--------|----|----|---------|--------|\n')
            
            for k in k_values:
                key = f"{method}_{k}"
                if key in all_results:
                    results = all_results[key].get('results', {})
                    for classifier in classifiers:
                        if classifier in results:
                            res = results[classifier]
                            f.write(f"| {k} | {classifier} | {res.get('accuracy', 0):.3f} | "
                                   f"{res.get('precision', 0):.3f} | {res.get('recall', 0):.3f} | "
                                   f"{res.get('f1', 0):.3f} | {res.get('auc', 0):.3f} | "
                                   f"{res.get('cv_mean', 0):.3f} | {res.get('cv_std', 0):.3f} |\n")
                else:
                    f.write(f"| {k} | - | - | - | - | - | - | - | - |\n")
            
            f.write('\n')
        
        # Лучшие результаты
        f.write('## Лучшие результаты\n\n')
        
        best_overall = None
        best_acc = 0
        
        for method in methods:
            for k in k_values:
                key = f"{method}_{k}"
                if key in all_results:
                    for classifier in classifiers:
                        if classifier in all_results[key].get('results', {}):
                            res = all_results[key]['results'][classifier]
                            acc = res.get('accuracy', 0)
                            if acc > best_acc:
                                best_acc = acc
                                best_overall = {
                                    'method': method,
                                    'k': k,
                                    'classifier': classifier,
                                    'results': res
                                }
        
        if best_overall:
            f.write(f"**Лучший результат**: {best_overall['method'].upper()}, K={best_overall['k']}, "
                   f"{best_overall['classifier']}\n\n")
            f.write(f"- Accuracy: {best_overall['results'].get('accuracy', 0):.3f}\n")
            f.write(f"- F1-score: {best_overall['results'].get('f1', 0):.3f}\n")
            f.write(f"- ROC AUC: {best_overall['results'].get('auc', 0):.3f}\n")
            f.write(f"- CV Mean: {best_overall['results'].get('cv_mean', 0):.3f} ± "
                   f"{best_overall['results'].get('cv_std', 0):.3f}\n\n")
        
        # Графики
        f.write('## Визуализации\n\n')
        f.write('![Accuracy по K](accuracy_by_k.png)\n\n')
        f.write('![F1-score по K](f1_by_k.png)\n\n')
        f.write('![ROC AUC по K](auc_by_k.png)\n\n')
        f.write('![Heatmap Accuracy](accuracy_heatmap.png)\n\n')
        
        # Выводы
        f.write('## Выводы\n\n')
        f.write('1. **Влияние количества ключевых слов (K)**:\n')
        f.write('   - Оптимальное значение K варьируется в зависимости от метода и классификатора\n')
        f.write('   - Как правило, больше ключевых слов (25-50) дают лучшие результаты\n')
        f.write('   - Слишком мало ключевых слов (5-10) может не хватать информации\n\n')
        
        f.write('2. **Сравнение методов извлечения**:\n')
        f.write('   - TF-IDF n-grams обычно показывает стабильные результаты\n')
        f.write('   - YAKE и TextRank могут давать разные результаты в зависимости от K\n\n')
        
        f.write('3. **Сравнение классификаторов**:\n')
        f.write('   - Все три классификатора показывают сопоставимые результаты\n')
        f.write('   - Random Forest может быть более устойчивым к переобучению\n\n')
        
        f.write('4. **Практические рекомендации**:\n')
        f.write('   - Использовать K=25-50 для лучшего баланса между точностью и скоростью\n')
        f.write('   - Комбинировать с семантическими методами (см. Эксперимент 2) для повышения точности\n')
        f.write('   - Ключевые слова могут быть полезны как дополнительный признак в ансамблевой системе\n\n')


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description='Эксперимент 4: Классификация на основе ключевых слов')
    parser.add_argument('--human_root', default='data/human', help='Корень папки с человеческими документами')
    parser.add_argument('--ai_root', default='data/ai', help='Корень папки с синтетическими документами')
    parser.add_argument('--output_dir', default='results/experiment4', help='Папка для результатов')
    parser.add_argument('--human_per_topic', type=int, default=50, help='Количество HUMAN документов на тему')
    parser.add_argument('--ai_per_model', type=int, default=100, help='Количество AI документов на модель')
    parser.add_argument('--models', nargs='+', default=['qwen', 'deepseek', 'gptoss'],
                       help='Модели AI для анализа')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Эксперимент 4: Классификация HUMAN vs AI на основе ключевых слов")
    print("=" * 60)
    
    # Загрузка документов
    print("\nЗагрузка документов...")
    human_docs = []
    for topic in ['text_mining', 'information_retrieval']:
        topic_dir = os.path.join(args.human_root, topic)
        if os.path.isdir(topic_dir):
            human_docs.extend(exp1.load_documents_from_txt_dir(topic_dir, count=args.human_per_topic))
    
    ai_docs = []
    for model in args.models:
        candidates = [
            f"{args.ai_root}/{model}_api_text/text_mining_full",
            f"{args.ai_root}/{model}_api_auto/text_mining_full",
            f"{args.ai_root}/{model}_api/text_mining_full",
            f"{args.ai_root}/{model}_api/ir",
        ]
        for cand in candidates:
            if len(ai_docs) >= args.ai_per_model * len(args.models):
                break
            left = args.ai_per_model * len(args.models) - len(ai_docs)
            ai_docs.extend(exp1.load_documents_from_txt_dir(cand, count=left))
    
    ai_docs = ai_docs[:args.ai_per_model * len(args.models)]
    
    print(f"HUMAN документов: {len(human_docs)}")
    print(f"AI документов: {len(ai_docs)}")
    
    if len(human_docs) == 0 or len(ai_docs) == 0:
        print("Ошибка: недостаточно документов для эксперимента")
        return
    
    # Запуск экспериментов
    methods = ['ngrams', 'yake', 'textrank']
    k_values = [5, 10, 25, 40, 50]
    
    all_results = {}
    
    print("\nЗапуск экспериментов...")
    for method in methods:
        print(f"\nМетод: {method.upper()}")
        for k in k_values:
            key = f"{method}_{k}"
            result = run_classification_experiment(human_docs, ai_docs, method, k)
            if result:
                all_results[key] = result
    
    # Визуализация
    print("\nСоздание графиков...")
    create_visualizations(all_results, args.output_dir)
    
    # Отчет
    print("\nГенерация отчета...")
    write_report(all_results, args.output_dir)
    
    # Сохранение JSON
    json_path = os.path.join(args.output_dir, 'experiment4_results.json')
    # Удаляем numpy массивы для JSON сериализации
    json_results = {}
    for key, val in all_results.items():
        json_results[key] = {
            'method': val.get('method'),
            'top_k': val.get('top_k'),
            'keywords': val.get('keywords', [])[:20],  # Только первые 20 для компактности
            'results': {}
        }
        for clf_name, clf_res in val.get('results', {}).items():
            json_results[key]['results'][clf_name] = {
                k: v for k, v in clf_res.items() 
                if k not in ['fpr', 'tpr']  # Исключаем массивы
            }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nГотово! Результаты сохранены в: {args.output_dir}")
    print(f"Отчет: {os.path.join(args.output_dir, 'experiment4_report.md')}")


if __name__ == '__main__':
    main()

