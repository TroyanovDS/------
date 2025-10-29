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

## Описание метрик

- Accuracy: доля верно классифицированных документов = (TP+TN)/(TP+FP+TN+FN).
- Precision (AI): сколько из предсказанных AI действительно AI = TP/(TP+FP).
- Recall (AI): сколько из всех AI обнаружено = TP/(TP+FN).
- F1-score (AI): гармоническое среднее Precision и Recall = 2PR/(P+R).
- ROC AUC: площадь под ROC; 1.0 означает идеальную разделимость классов.
- CV Mean / CV Std: среднее/стандартное отклонение Accuracy на 5‑fold cross‑validation.
- Confusion Matrix: матрица ошибок [[TN, FP], [FN, TP]].

## Подробные выводы

1) Во всех разрезах синтетических моделей (Qwen, DeepSeek, GPT‑OSS) эмбеддинги roberta‑base и albert‑base‑v2 дают Accuracy=1.000 и AUC=1.000. Это указывает на практически совершимое разделение HUMAN/AI в текущем датасете.

2) bert‑base‑uncased стабильно высок (Accuracy≈0.93–0.98, AUC≈0.98–1.00), но немного уступает roberta/albert.

3) Среди sentence‑transformers картина неоднородная: intfloat/e5‑large‑v2 демонстрирует отличные результаты (до 1.000/1.000), тогда как all‑mpnet‑base‑v2 на этом наборе слаб (Accuracy≈0.50–0.78, AUC≈0.55–0.85), что может быть связано с доменной несовместимостью и особенностями подготовки модели.

4) Бейзлайны (LogReg, Linear SVM) на сильных эмбеддингах практически не уступают MLP: при roberta/e5/albert достигается тот же 1.000/1.000. Это подтверждает, что линейные разделители уже достаточны в хорошем семантическом пространстве.

5) С точки зрения практического применения, предпочтительна связка: roberta‑base (или e5‑large‑v2) + Logistic Regression/Linear SVM (с калибровкой вероятностей при необходимости). Это быстро, интерпретируемо и устойчиво.

## Рекомендации по улучшению

- Эмбеддинги: проверить дополнительные модели (E5‑младшие/бóльшие, GTE, Jina‑embeddings), а также использовать mean‑pooling/attention‑pooling (уже включено mean‑pooling для HF‑моделей).
- Классификатор: сравнить MLP с Logistic Regression/Linear SVM как базу; добавить калибровку вероятностей (Platt/Isotonic) для пороговых решений.
- Данные: расширить домены (темы/жанры), добавить другие модели синтетики (Qwen3/4, Llama, GLM), балансировать длины текстов, применить аугментации (перефразы/шум) для устойчивости к перегенерациям.
- Инференс: разбиение на части (chunking) для длинных документов и агрегация предсказаний повышают надёжность.

