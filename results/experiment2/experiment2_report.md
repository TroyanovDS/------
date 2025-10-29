# Эксперимент 2: Детекция AI-текстов с помощью эмбеддингов (пер‑модельный разрез)

## Методология

- **Корпуса**: 100 HUMAN (50 TM + 50 IR) против 100 AI на каждую синтетическую модель
- **Эмбеддинги**: bert-base-uncased, roberta-base, albert-base-v2, sentence-transformers/all-mpnet-base-v2, intfloat/e5-large-v2
- **Классификатор**: MLP, train/test + 5‑fold CV

## Модель синтетики: QWEN

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.925 | 0.978 | 0.975 | 0.036 |
| bert-base-uncased/logreg | 0.975 | 1.000 | 0.994 | 0.013 |
| bert-base-uncased/linear_svm | 0.975 | 1.000 | 0.994 | 0.013 |
| roberta-base | 1.000 | 1.000 | 1.000 | 0.000 |
| roberta-base/logreg | 1.000 | 1.000 | 1.000 | 0.000 |
| roberta-base/linear_svm | 1.000 | 1.000 | 1.000 | 0.000 |
| albert-base-v2 | 1.000 | 1.000 | 0.975 | 0.036 |
| albert-base-v2/logreg | 1.000 | 1.000 | 0.988 | 0.025 |
| albert-base-v2/linear_svm | 1.000 | 1.000 | 0.988 | 0.025 |
| sentence-transformers/all-mpnet-base-v2 | 0.575 | 0.627 | 0.688 | 0.071 |
| sentence-transformers/all-mpnet-base-v2/logreg | 0.725 | 0.850 | 0.831 | 0.061 |
| sentence-transformers/all-mpnet-base-v2/linear_svm | 0.775 | 0.853 | 0.825 | 0.064 |
| intfloat/e5-large-v2 | 0.700 | 0.785 | 0.838 | 0.050 |
| intfloat/e5-large-v2/logreg | 0.900 | 0.953 | 0.894 | 0.032 |
| intfloat/e5-large-v2/linear_svm | 0.900 | 0.958 | 0.875 | 0.034 |

![Метрики](classification_metrics_qwen.png)

![ROC](roc_curves_qwen.png)

![Матрицы ошибок](confusion_matrices_qwen.png)

## Модель синтетики: DEEPSEEK

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.975 | 1.000 | 0.981 | 0.025 |
| bert-base-uncased/logreg | 1.000 | 1.000 | 0.988 | 0.015 |
| bert-base-uncased/linear_svm | 1.000 | 1.000 | 0.988 | 0.015 |
| roberta-base | 1.000 | 1.000 | 1.000 | 0.000 |
| roberta-base/logreg | 1.000 | 1.000 | 1.000 | 0.000 |
| roberta-base/linear_svm | 1.000 | 1.000 | 1.000 | 0.000 |
| albert-base-v2 | 0.975 | 1.000 | 0.994 | 0.013 |
| albert-base-v2/logreg | 0.950 | 1.000 | 0.994 | 0.013 |
| albert-base-v2/linear_svm | 0.950 | 1.000 | 1.000 | 0.000 |
| sentence-transformers/all-mpnet-base-v2 | 0.500 | 0.547 | 0.544 | 0.070 |
| sentence-transformers/all-mpnet-base-v2/logreg | 0.825 | 0.907 | 0.725 | 0.036 |
| sentence-transformers/all-mpnet-base-v2/linear_svm | 0.825 | 0.913 | 0.719 | 0.056 |
| intfloat/e5-large-v2 | 1.000 | 1.000 | 1.000 | 0.000 |
| intfloat/e5-large-v2/logreg | 1.000 | 1.000 | 1.000 | 0.000 |
| intfloat/e5-large-v2/linear_svm | 1.000 | 1.000 | 1.000 | 0.000 |

![Метрики](classification_metrics_deepseek.png)

![ROC](roc_curves_deepseek.png)

![Матрицы ошибок](confusion_matrices_deepseek.png)

## Модель синтетики: GPTOSS

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.975 | 0.997 | 0.969 | 0.000 |
| bert-base-uncased/logreg | 1.000 | 1.000 | 0.975 | 0.012 |
| bert-base-uncased/linear_svm | 1.000 | 1.000 | 0.975 | 0.012 |
| roberta-base | 1.000 | 1.000 | 0.981 | 0.025 |
| roberta-base/logreg | 1.000 | 1.000 | 1.000 | 0.000 |
| roberta-base/linear_svm | 1.000 | 1.000 | 1.000 | 0.000 |
| albert-base-v2 | 1.000 | 1.000 | 0.975 | 0.023 |
| albert-base-v2/logreg | 1.000 | 1.000 | 0.981 | 0.015 |
| albert-base-v2/linear_svm | 1.000 | 1.000 | 0.981 | 0.015 |
| sentence-transformers/all-mpnet-base-v2 | 0.550 | 0.630 | 0.613 | 0.083 |
| sentence-transformers/all-mpnet-base-v2/logreg | 0.725 | 0.833 | 0.713 | 0.072 |
| sentence-transformers/all-mpnet-base-v2/linear_svm | 0.750 | 0.835 | 0.719 | 0.066 |
| intfloat/e5-large-v2 | 0.975 | 1.000 | 0.981 | 0.025 |
| intfloat/e5-large-v2/logreg | 1.000 | 1.000 | 0.988 | 0.015 |
| intfloat/e5-large-v2/linear_svm | 1.000 | 1.000 | 0.988 | 0.015 |

![Метрики](classification_metrics_gptoss.png)

![ROC](roc_curves_gptoss.png)

![Матрицы ошибок](confusion_matrices_gptoss.png)

