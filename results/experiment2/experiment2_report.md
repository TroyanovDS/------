# Эксперимент 2: Детекция AI-текстов с помощью эмбеддингов (пер‑модельный разрез)

## Методология

- **Корпуса**: 100 HUMAN (50 TM + 50 IR) против 100 AI на каждую синтетическую модель
- **Эмбеддинги**: bert-base-uncased, roberta-base, albert-base-v2
- **Классификатор**: MLP, train/test + 5‑fold CV

## Модель синтетики: QWEN

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.925 | 0.987 | 0.963 | 0.046 |
| roberta-base | 1.000 | 1.000 | 0.994 | 0.012 |
| albert-base-v2 | 1.000 | 1.000 | 0.988 | 0.025 |

![Метрики](classification_metrics_qwen.png)

![ROC](roc_curves_qwen.png)

![Матрицы ошибок](confusion_matrices_qwen.png)

## Модель синтетики: DEEPSEEK

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.950 | 0.980 | 0.944 | 0.041 |
| roberta-base | 1.000 | 1.000 | 0.994 | 0.012 |
| albert-base-v2 | 0.975 | 1.000 | 1.000 | 0.000 |

![Метрики](classification_metrics_deepseek.png)

![ROC](roc_curves_deepseek.png)

![Матрицы ошибок](confusion_matrices_deepseek.png)

## Модель синтетики: GPTOSS

### Сводная таблица по эмбеддингам

| Embedding | Accuracy | AUC | CV Mean | CV Std |
|-----------|----------|-----|---------|--------|
| bert-base-uncased | 0.975 | 0.995 | 0.975 | 0.023 |
| roberta-base | 1.000 | 1.000 | 1.000 | 0.000 |
| albert-base-v2 | 0.950 | 1.000 | 0.981 | 0.015 |

![Метрики](classification_metrics_gptoss.png)

![ROC](roc_curves_gptoss.png)

![Матрицы ошибок](confusion_matrices_gptoss.png)

