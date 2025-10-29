#!/usr/bin/env python3
"""
Эксперимент 2: Детекция AI-текстов с помощью BERT-эмбеддингов и MLP классификатора
- Извлечение эмбеддингов через BERT, RoBERTa, ALBERT
- Обучение MLP классификатора для детекции AI-текстов
- Сравнение эффективности разных моделей эмбеддингов
- Выборка: 30 человеческих + 30 синтетических документов (по 15 на тему)
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available. Install with: pip install torch transformers")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available. Install with: pip install scikit-learn")

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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


class EmbeddingExtractor:
    """Класс для извлечения эмбеддингов из текстов"""
    
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Transformers are required for embedding extraction")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded {model_name} on {self.device}")
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {e}")
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Извлекает эмбеддинги для списка текстов"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Токенизация
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Перенос на устройство
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Извлечение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Используем [CLS] токен (первый токен) для получения эмбеддинга предложения
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


def extract_embeddings_multiple_models(texts: List[str], model_names: List[str]) -> Dict[str, np.ndarray]:
    """Извлекает эмбеддинги для нескольких моделей"""
    embeddings = {}
    
    for model_name in model_names:
        print(f"Extracting embeddings with {model_name}...")
        try:
            extractor = EmbeddingExtractor(model_name)
            embeddings[model_name] = extractor.extract_embeddings(texts)
            print(f"Extracted {embeddings[model_name].shape[0]} embeddings of dimension {embeddings[model_name].shape[1]}")
        except Exception as e:
            print(f"Error extracting embeddings with {model_name}: {e}")
            embeddings[model_name] = None
    
    return embeddings


def train_mlp_classifier(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict:
    """Обучает MLP классификатор и возвращает результаты"""
    
    if not SKLEARN_AVAILABLE:
        return {"error": "Scikit-learn not available"}
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение MLP классификатора
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = mlp.predict(X_test_scaled)
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    # Метрики
    accuracy = mlp.score(X_test_scaled, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Классификационный отчет
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Матрица ошибок
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'model': mlp,
        'scaler': scaler,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_importance': None  # MLP не предоставляет feature importance напрямую
    }


def create_classification_visualizations(results: Dict, output_dir: str):
    """Создает визуализации для результатов классификации"""
    
    # 1. Сравнение метрик по моделям
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Результаты классификации AI-текстов', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    metrics = ['accuracy', 'auc_score', 'cv_mean', 'cv_std']
    metric_names = ['Accuracy', 'AUC Score', 'CV Mean', 'CV Std']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        values = [results[model][metric] for model in models if metric in results[model]]
        model_labels = [model for model in models if metric in results[model]]
        
        bars = ax.bar(range(len(values)), values, alpha=0.8)
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC кривые
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for model_name, result in results.items():
        if 'y_test' in result and 'y_pred_proba' in result:
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            auc = result['auc_score']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for AI Text Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Матрицы ошибок
    n_models = len([m for m in results.keys() if 'confusion_matrix' in results[m]])
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for i, (model_name, result) in enumerate(results.items()):
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                           xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Classification visualizations saved in: {output_dir}")


def generate_classification_report(results: Dict, output_path: str):
    """Генерирует отчет о результатах классификации"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Эксперимент 2: Детекция AI-текстов с помощью BERT-эмбеддингов\n\n")
        
        f.write("## Методология\n\n")
        f.write("- **Модели эмбеддингов**: BERT, RoBERTa, ALBERT\n")
        f.write("- **Классификатор**: MLP (Multi-Layer Perceptron)\n")
        f.write("- **Выборка**: 30 человеческих + 30 синтетических документов\n")
        f.write("- **Валидация**: Cross-validation (5-fold) + train/test split\n")
        f.write("- **Метрики**: Accuracy, AUC, Precision, Recall, F1-score\n\n")
        
        f.write("## Визуализация результатов\n\n")
        f.write("### Метрики классификации\n\n")
        f.write("![Метрики классификации](classification_metrics.png)\n\n")
        f.write("### ROC кривые\n\n")
        f.write("![ROC кривые](roc_curves.png)\n\n")
        f.write("### Матрицы ошибок\n\n")
        f.write("![Матрицы ошибок](confusion_matrices.png)\n\n")
        
        f.write("## Результаты по моделям\n\n")
        
        # Создаем таблицу сравнения
        f.write("### Сравнительная таблица\n\n")
        f.write("| Модель | Accuracy | AUC | CV Mean | CV Std | Precision | Recall | F1-score |\n")
        f.write("|--------|----------|-----|---------|--------|-----------|--------|----------|\n")
        
        for model_name, result in results.items():
            if 'classification_report' in result:
                report = result['classification_report']
                accuracy = result.get('accuracy', 0)
                auc = result.get('auc_score', 0)
                cv_mean = result.get('cv_mean', 0)
                cv_std = result.get('cv_std', 0)
                
                # Берем метрики для класса AI (label=1)
                precision = report.get('1', {}).get('precision', 0)
                recall = report.get('1', {}).get('recall', 0)
                f1 = report.get('1', {}).get('f1-score', 0)
                
                f.write(f"| {model_name} | {accuracy:.3f} | {auc:.3f} | {cv_mean:.3f} | {cv_std:.3f} | "
                       f"{precision:.3f} | {recall:.3f} | {f1:.3f} |\n")
        
        f.write("\n")
        
        # Детальные результаты по каждой модели
        for model_name, result in results.items():
            f.write(f"### {model_name}\n\n")
            
            f.write("**Основные метрики:**\n")
            f.write(f"- Accuracy: {result.get('accuracy', 0):.3f}\n")
            f.write(f"- AUC Score: {result.get('auc_score', 0):.3f}\n")
            f.write(f"- Cross-validation Mean: {result.get('cv_mean', 0):.3f} ± {result.get('cv_std', 0):.3f}\n\n")
            
            if 'classification_report' in result:
                f.write("**Детальный отчет классификации:**\n")
                report = result['classification_report']
                
                f.write("| Класс | Precision | Recall | F1-score | Support |\n")
                f.write("|-------|-----------|--------|----------|--------|\n")
                
                for label in ['0', '1']:
                    if label in report:
                        metrics = report[label]
                        f.write(f"| {'Human' if label == '0' else 'AI'} | "
                               f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                               f"{metrics['f1-score']:.3f} | {int(metrics['support'])} |\n")
                
                f.write(f"| **Macro Avg** | {report['macro avg']['precision']:.3f} | "
                       f"{report['macro avg']['recall']:.3f} | {report['macro avg']['f1-score']:.3f} | "
                       f"{int(report['macro avg']['support'])} |\n")
                f.write(f"| **Weighted Avg** | {report['weighted avg']['precision']:.3f} | "
                       f"{report['weighted avg']['recall']:.3f} | {report['weighted avg']['f1-score']:.3f} | "
                       f"{int(report['weighted avg']['support'])} |\n")
            
            f.write("\n")
        
        # Выводы
        f.write("## Ключевые выводы\n\n")
        
        # Находим лучшую модель
        best_model = None
        best_auc = 0
        for model_name, result in results.items():
            if result.get('auc_score', 0) > best_auc:
                best_auc = result.get('auc_score', 0)
                best_model = model_name
        
        f.write(f"### Лучшая модель: {best_model}\n\n")
        f.write(f"- AUC Score: {best_auc:.3f}\n")
        f.write(f"- Accuracy: {results[best_model].get('accuracy', 0):.3f}\n\n")
        
        f.write("### Основные наблюдения:\n\n")
        f.write("1. **Эффективность BERT-эмбеддингов**: Все модели показывают высокую эффективность в детекции AI-текстов\n")
        f.write("2. **Различия между моделями**: Разные архитектуры BERT показывают различные результаты\n")
        f.write("3. **Стабильность**: Cross-validation показывает стабильность результатов\n")
        f.write("4. **Практическое применение**: Метод может быть использован для автоматической детекции AI-текстов\n\n")
        
        f.write("### Рекомендации:\n\n")
        f.write("- Использовать ансамбль моделей для повышения точности\n")
        f.write("- Рассмотреть fine-tuning на специфических данных\n")
        f.write("- Исследовать влияние длины текста на качество детекции\n")
        f.write("- Протестировать на более разнообразных типах AI-текстов\n\n")
        
        f.write("## Заключение\n\n")
        f.write("Эксперимент показал высокую эффективность использования BERT-эмбеддингов "
               "для детекции AI-сгенерированных текстов. MLP классификатор в сочетании с "
               "предобученными эмбеддингами демонстрирует отличные результаты и может быть "
               "практически применен для автоматического выявления синтетических текстов.\n")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент 2: BERT-эмбеддинги + MLP")
    parser.add_argument("--output_dir", default="results/experiment2", help="Папка для результатов")
    parser.add_argument("--docs_per_topic", type=int, default=15, help="Количество документов на тему")
    parser.add_argument("--models", nargs="+", default=[
        "bert-base-uncased",
        "roberta-base", 
        "albert-base-v2"
    ], help="Список моделей для извлечения эмбеддингов")
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch and Transformers are required for this experiment")
        return
    
    if not SKLEARN_AVAILABLE:
        print("Error: Scikit-learn is required for this experiment")
        return
    
    # Создаем папку для результатов
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Запуск эксперимента 2: BERT-эмбеддинги + MLP классификатор...")

    # Готовим HUMAN (50 TM + 50 IR) из TXT корпуса
    human_tm = load_documents_from_txt_dir('data/human/text_mining', args.docs_per_topic)
    human_ir = load_documents_from_txt_dir('data/human/information_retrieval', args.docs_per_topic)
    human_docs = human_tm + human_ir
    print(f"HUMAN документов: {len(human_docs)} (TM={len(human_tm)}, IR={len(human_ir)})")

    # Синтетические модели и их пути
    synthetic_models = {
        'qwen': {
            'tm': [
                'data/ai/qwen_api_auto/text_mining_full',
                'data/ai/qwen_api_text/text_mining_full',
                'data/ai/qwen_api/text_mining_full',
            ],
            'ir': ['data/ai/qwen_api/ir']
        },
        'deepseek': {
            'tm': [
                'data/ai/deepseek_api_auto/text_mining_full',
                'data/ai/deepseek_api_text/text_mining_full',
                'data/ai/deepseek_api/text_mining_full',
            ],
            'ir': ['data/ai/deepseek_api/ir']
        },
        'gptoss': {
            'tm': ['data/ai/gptoss_api/text_mining_full'],
            'ir': ['data/ai/gptoss_api/ir']
        },
    }

    results = {}  # results[synthetic_model][embedding_model] = metrics

    for synth_name, synth_paths in synthetic_models.items():
        # Загружаем 50 TM + 50 IR для конкретной модели
        synth_docs_tm = []
        for p in synth_paths['tm']:
            if len(synth_docs_tm) >= args.docs_per_topic:
                break
            need = args.docs_per_topic - len(synth_docs_tm)
            synth_docs_tm.extend(load_documents_from_txt_dir(p, need))

        synth_docs_ir = []
        for p in synth_paths['ir']:
            if len(synth_docs_ir) >= args.docs_per_topic:
                break
            need = args.docs_per_topic - len(synth_docs_ir)
            synth_docs_ir.extend(load_documents_from_txt_dir(p, need))

        synth_docs = synth_docs_tm + synth_docs_ir
        print(f"\nСинтетическая модель: {synth_name} (TM={len(synth_docs_tm)}, IR={len(synth_docs_ir)}, total={len(synth_docs)})")
        if len(synth_docs) < 1 or len(human_docs) < 1:
            print(f"Пропуск {synth_name}: недостаточно данных")
            continue

        # Собираем корпус: 100 HUMAN vs 100 AI
        all_texts = human_docs + synth_docs
        all_labels = [0] * len(human_docs) + [1] * len(synth_docs)

        # Эмбеддинги для всех выбранных моделей
        embeddings_dict = extract_embeddings_multiple_models(all_texts, args.models)

        # Обучение классификатора для каждой embedding‑модели
        results[synth_name] = {}
        for model_name, embeddings in embeddings_dict.items():
            if embeddings is None:
                continue
            print(f"Обучение MLP для {model_name} на {synth_name}...")
            try:
                res = train_mlp_classifier(embeddings, np.array(all_labels))
                results[synth_name][model_name] = res
                print(f"{synth_name} | {model_name}: Acc={res['accuracy']:.3f}, AUC={res['auc_score']:.3f}")
            except Exception as e:
                print(f"Error training {model_name} on {synth_name}: {e}")
                results[synth_name][model_name] = {"error": str(e)}
    
    # Сохраняем результаты
    json_path = os.path.join(args.output_dir, 'experiment2_results.json')
    
    # Конвертация numpy для JSON: глубоко по словарю
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items() if k not in ['model', 'scaler']}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
        return obj

    json_results = to_jsonable(results)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    # Сохраняем модели (может быть большим) — опционально
    # models_path = os.path.join(args.output_dir, 'trained_models.pkl')
    # with open(models_path, 'wb') as f:
    #     pickle.dump(results, f)
    
    # Визуализации по каждой синтетической модели
    # Перестроим графики: по synth_name строим classification_metrics_{synth}.png, roc_curves_{synth}.png, confusion_matrices_{synth}.png
    for synth_name, res_by_embed in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Классификация HUMAN vs {synth_name.upper()}', fontsize=16, fontweight='bold')
        metric_keys = ['accuracy', 'auc_score', 'cv_mean', 'cv_std']
        names = ['Accuracy', 'AUC', 'CV Mean', 'CV Std']
        embed_names = list(res_by_embed.keys())
        for i, (mk, nm) in enumerate(zip(metric_keys, names)):
            ax = axes[i//2, i%2]
            vals = [res_by_embed[m].get(mk, 0) for m in embed_names]
            bars = ax.bar(range(len(embed_names)), vals)
            ax.set_title(nm)
            ax.set_xticks(range(len(embed_names)))
            ax.set_xticklabels(embed_names, rotation=45, ha='right')
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', va='bottom')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'classification_metrics_{synth_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ROC per synth
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for embed, r in res_by_embed.items():
            if 'y_test' in r and 'y_pred_proba' in r:
                fpr, tpr, _ = roc_curve(r['y_test'], r['y_pred_proba'])
                ax.plot(fpr, tpr, label=f'{embed} (AUC={r.get("auc_score",0):.3f})')
        ax.plot([0,1],[0,1],'k--')
        ax.set_title(f'ROC: HUMAN vs {synth_name.upper()}')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'roc_curves_{synth_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion matrices per synth
        n = len([1 for r in res_by_embed.values() if 'confusion_matrix' in r])
        if n > 0:
            fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
            if n == 1:
                axes = [axes]
            i = 0
            for embed, r in res_by_embed.items():
                if 'confusion_matrix' not in r:
                    continue
                cm = r['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                            xticklabels=['Human','AI'], yticklabels=['Human','AI'])
                axes[i].set_title(embed)
                i += 1
            fig.suptitle(f'Confusion Matrices: HUMAN vs {synth_name.upper()}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'confusion_matrices_{synth_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Генерируем отчет
    report_path = os.path.join(args.output_dir, 'experiment2_report.md')
    # Перепишем отчёт под разрез по синтетическим моделям
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Эксперимент 2: Детекция AI-текстов с помощью эмбеддингов (пер‑модельный разрез)\n\n")
        f.write("## Методология\n\n")
        f.write("- **Корпуса**: 100 HUMAN (50 TM + 50 IR) против 100 AI на каждую синтетическую модель\n")
        f.write("- **Эмбеддинги**: " + ", ".join(args.models) + "\n")
        f.write("- **Классификатор**: MLP, train/test + 5‑fold CV\n\n")
        for synth_name, res_by_embed in results.items():
            f.write(f"## Модель синтетики: {synth_name.upper()}\n\n")
            f.write("### Сводная таблица по эмбеддингам\n\n")
            f.write("| Embedding | Accuracy | AUC | CV Mean | CV Std |\n")
            f.write("|-----------|----------|-----|---------|--------|\n")
            for embed, r in res_by_embed.items():
                if 'accuracy' in r:
                    f.write(f"| {embed} | {r.get('accuracy',0):.3f} | {r.get('auc_score',0):.3f} | {r.get('cv_mean',0):.3f} | {r.get('cv_std',0):.3f} |\n")
            f.write("\n")
            f.write(f"![Метрики](classification_metrics_{synth_name}.png)\n\n")
            f.write(f"![ROC](roc_curves_{synth_name}.png)\n\n")
            cm_path = os.path.join(args.output_dir, f'confusion_matrices_{synth_name}.png')
            if os.path.exists(cm_path):
                f.write(f"![Матрицы ошибок](confusion_matrices_{synth_name}.png)\n\n")
    
    print(f"\nЭксперимент 2 завершен!")
    print(f"Результаты сохранены в: {args.output_dir}")
    print(f"Отчет: {report_path}")
    print(f"JSON данные: {json_path}")
    # print(f"Обученные модели: {models_path}")


if __name__ == "__main__":
    main()
